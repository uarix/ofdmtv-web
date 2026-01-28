/*
OFDM TV Streaming Decoder for WebAssembly
Real-time sample-by-sample decoding with line callbacks
*/

#pragma once

#include <vector>
#include <string>
#include <functional>
#include <cmath>

namespace DSP { using std::abs; using std::min; using std::cos; using std::sin; }

#include "complex.hh"
#include "utils.hh"
#include "fft.hh"
#include "mls.hh"
#include "crc.hh"
#include "bip_buffer.hh"
#include "theil_sen.hh"
#include "trigger.hh"
#include "blockdc.hh"
#include "hilbert.hh"
#include "phasor.hh"
#include "delay.hh"
#include "sma.hh"
#include "osd.hh"

typedef float value;
typedef DSP::Complex<value> cmplx;

// Schmidl-Cox synchronization detector
template <typename value, typename cmplx, int search_pos, int symbol_len, int guard_len>
struct StreamingSchmidlCox
{
    typedef DSP::Const<value> Const;
    static const int match_len = guard_len | 1;
    static const int match_del = (match_len - 1) / 2;
    DSP::FastFourierTransform<symbol_len, cmplx, -1> fwd;
    DSP::FastFourierTransform<symbol_len, cmplx, 1> bwd;
    DSP::SMA4<cmplx, value, symbol_len, false> cor;
    DSP::SMA4<value, value, 2*symbol_len, false> pwr;
    DSP::SMA4<value, value, match_len, false> match;
    DSP::Delay<value, match_del> delay;
    DSP::SchmittTrigger<value> threshold;
    DSP::FallingEdgeTrigger falling;
    cmplx tmp0[symbol_len], tmp1[symbol_len], tmp2[symbol_len];
    cmplx seq[symbol_len], kern[symbol_len];
    cmplx cmplx_shift = 0;
    value timing_max = 0;
    value phase_max = 0;
    int index_max = 0;

    static int bin(int carrier)
    {
        return (carrier + symbol_len) % symbol_len;
    }
public:
    int symbol_pos = 0;
    value cfo_rad = 0;
    value frac_cfo = 0;

    StreamingSchmidlCox(const cmplx *sequence) : threshold(value(0.17*match_len), value(0.19*match_len))
    {
        for (int i = 0; i < symbol_len; ++i)
            seq[i] = sequence[i];
        fwd(kern, sequence);
        for (int i = 0; i < symbol_len; ++i)
            kern[i] = conj(kern[i]) / value(symbol_len);
    }
    
    bool operator()(const cmplx *samples)
    {
        cmplx P = cor(samples[search_pos+symbol_len] * conj(samples[search_pos+2*symbol_len]));
        value R = value(0.5) * pwr(norm(samples[search_pos+2*symbol_len]));
        value min_R = 0.0001 * symbol_len;
        R = std::max(R, min_R);
        value timing = match(norm(P) / (R * R));
        value phase = delay(arg(P));

        bool collect = threshold(timing);
        bool process = falling(collect);

        if (!collect && !process)
            return false;

        if (timing_max < timing) {
            timing_max = timing;
            phase_max = phase;
            index_max = match_del;
        } else if (index_max < symbol_len + guard_len + match_del) {
            ++index_max;
        }

        if (!process)
            return false;

        frac_cfo = phase_max / value(symbol_len);

        DSP::Phasor<cmplx> osc;
        osc.omega(frac_cfo);
        symbol_pos = search_pos - index_max;
        index_max = 0;
        timing_max = 0;
        for (int i = 0; i < symbol_len; ++i)
            tmp1[i] = samples[i+symbol_pos+symbol_len] * osc();
        fwd(tmp0, tmp1);
        for (int i = 0; i < symbol_len; ++i)
            tmp0[i] *= kern[i];
        bwd(tmp2, tmp0);
        int shift = cyclic_prefix_length(tmp2);
        cmplx_shift = tmp2[shift];
        value finer_cfo = arg(cmplx_shift) / value(symbol_len);
        cfo_rad = frac_cfo + finer_cfo;
        return true;
    }

    int cyclic_prefix_length(const cmplx *tmp)
    {
        value max_val = 0;
        int max_idx = 0;
        for (int i = 0; i < symbol_len; ++i) {
            value val = norm(tmp[i]);
            if (val > max_val) {
                max_val = val;
                max_idx = i;
            }
        }
        return max_idx >= symbol_len/2 ? max_idx - symbol_len : max_idx;
    }
};

// Line callback type: (lineNumber, numLines, rgbData[numLines * 320 * 3])
using LineCallback = std::function<void(int, int, const uint8_t*)>;
// Status callback type: (status message, callsign)  
using StatusCallback = std::function<void(const std::string&, const std::string&)>;

// Streaming decoder states
enum class DecoderState {
    WAITING_FOR_SYNC,       // Waiting for Schmidl-Cox sync
    DECODING_METADATA,      // Decoding call sign and mode
    DECODING_IMAGE,         // Decoding image lines
    COMPLETE,               // Frame complete
    ERROR                   // Decode error
};

template <int rate>
class StreamingDecoder
{
public:
    typedef DSP::Const<value> Const;
    static const int symbol_len = (1280 * rate) / 8000;
    static const int filter_len = (((21 * rate) / 8000) & ~3) | 1;
    static const int guard_len = symbol_len / 8;
    static const int img_width = 320;
    static const int img_height = 240;
    static const int teeth_count = 16;
    static const int teeth_dist = img_width / teeth_count;
    static const int teeth_off = teeth_dist / 2;
    static const int mls0_len = 127;
    static const int mls0_off = -mls0_len + 1;
    static const int mls0_poly = 0b10001001;
    static const int mls1_len = img_width + teeth_count;
    static const int mls1_off = -mls1_len / 2;
    static const int mls1_poly = 0b1100110001;
    static const int mls2_poly = 0b10001000000001011;
    static const int mls3_poly = 0b10111010010000001;
    static const int mls4_len = 255;
    static const int mls4_off = -mls4_len / 2;
    static const int mls4_poly = 0b100101011;
    static const int buffer_len = 6 * (symbol_len + guard_len);
    static const int search_pos = buffer_len - 4 * (symbol_len + guard_len);

private:
    // DSP components
    DSP::FastFourierTransform<symbol_len, cmplx, -1> fwd;
    DSP::FastFourierTransform<symbol_len, cmplx, 1> bwd;
    DSP::BlockDC<value, value> blockdc;
    DSP::Hilbert<cmplx, filter_len> hilbert;
    DSP::BipBuffer<cmplx, buffer_len> input_hist;
    DSP::TheilSenEstimator<value, mls1_len> tse;
    StreamingSchmidlCox<value, cmplx, search_pos, symbol_len/2, guard_len> correlator;
    CODE::CRC<uint16_t> crc;
    CODE::OrderedStatisticsDecoder<255, 71, 4> osddec;
    
    // Oscillator for frequency correction
    DSP::Phasor<cmplx> osc;
    
    // Buffers
    int8_t genmat[255*71];
    cmplx chan[mls1_len];
    cmplx fdom[2 * symbol_len], tdom[symbol_len];
    value rgb_line[2 * 3 * img_width];
    value phase[mls1_len], index[mls1_len];
    
    // State
    DecoderState state;
    int symbol_pos;
    value cfo_rad;
    int current_line;
    int samples_in_symbol;
    int post_sync_samples;
    std::string call_sign;
    
    // MLS sequences for image decoding
    CODE::MLS *seq0, *seq1, *seq2, *seq3;
    
    // Callbacks
    LineCallback lineCallback;
    StatusCallback statusCallback;
    
    // Helper
    static int bin(int carrier) {
        return (carrier + symbol_len) % symbol_len;
    }
    static value nrz(bool bit) {
        return 1 - 2 * bit;
    }
    
    const cmplx* mls0_seq() {
        CODE::MLS seq0_init(mls0_poly);
        static cmplx seq_buf[symbol_len];
        for (int i = 0; i < symbol_len/2; ++i)
            seq_buf[i] = 0;
        for (int i = 0; i < mls0_len; ++i)
            seq_buf[(i+mls0_off/2+symbol_len/2)%(symbol_len/2)] = nrz(seq0_init());
        return seq_buf;
    }
    
    void yuv_to_rgb(value *rgb, const value *yuv) {
        value WR(0.299), WB(0.114), WG(1-WR-WB), UMAX(0.493), VMAX(0.877);
        rgb[0] = yuv[0] + ((1-WR)/VMAX) * yuv[2];
        rgb[1] = yuv[0] - (WB*(1-WB)/(UMAX*WG)) * yuv[1] - (WR*(1-WR)/(VMAX*WG)) * yuv[2];
        rgb[2] = yuv[0] + ((1-WB)/UMAX) * yuv[1];
    }
    
    void cmplx_to_rgb(value *rgb0, value *rgb1, cmplx *inp0, cmplx *inp1) {
        value upv[2] = {
            inp0[0].real()-inp0[0].imag(),
            inp1[1].real()-inp1[1].imag()
        };
        value umv[2] = {
            inp1[0].real()-inp1[0].imag(),
            inp0[1].real()-inp0[1].imag()
        };
        value yuv0[6] = {
            (inp0[0].real()+inp0[0].imag()+umv[0])/2, (upv[0]+umv[0])/2, (upv[0]-umv[0])/2,
            (upv[1]-inp0[1].real()-inp0[1].imag())/2, (upv[1]+umv[1])/2, (upv[1]-umv[1])/2
        };
        value yuv1[6] = {
            (upv[0]-inp1[0].real()-inp1[0].imag())/2, (upv[0]+umv[0])/2, (upv[0]-umv[0])/2,
            (inp1[1].real()+inp1[1].imag()+umv[1])/2, (upv[1]+umv[1])/2, (upv[1]-umv[1])/2
        };
        yuv_to_rgb(rgb0, yuv0);
        yuv_to_rgb(rgb1, yuv1);
        yuv_to_rgb(rgb0+3, yuv0+3);
        yuv_to_rgb(rgb1+3, yuv1+3);
    }
    
    static void base37_decoder(char *str, long long int val, int len) {
        for (int i = len - 1; i >= 0; --i) {
            int mod = val % 37;
            if (mod == 0)
                str[i] = ' ';
            else if (mod <= 10)
                str[i] = '0' + mod - 1;
            else
                str[i] = 'A' + mod - 11;
            val /= 37;
        }
    }

public:
    StreamingDecoder() : 
        correlator(mls0_seq()), 
        crc(0xA8F4),
        state(DecoderState::WAITING_FOR_SYNC),
        symbol_pos(0),
        cfo_rad(0),
        current_line(0),
        samples_in_symbol(0),
        post_sync_samples(0),
        seq0(nullptr), seq1(nullptr), seq2(nullptr), seq3(nullptr)
    {
        CODE::BoseChaudhuriHocquenghemGenerator<255, 71>::matrix(genmat, true, {
            0b100011101, 0b101110111, 0b111110011, 0b101101001,
            0b110111101, 0b111100111, 0b100101011, 0b111010111,
            0b000010011, 0b101100101, 0b110001011, 0b101100011,
            0b100011011, 0b100111111, 0b110001101, 0b100101101,
            0b101011111, 0b111111001, 0b111000011, 0b100111001,
            0b110101001, 0b000011111, 0b110000111, 0b110110001});
        blockdc.samples(2*(symbol_len+guard_len));
    }
    
    ~StreamingDecoder() {
        delete seq0;
        delete seq1;
        delete seq2;
        delete seq3;
    }
    
    void setLineCallback(LineCallback cb) { lineCallback = cb; }
    void setStatusCallback(StatusCallback cb) { statusCallback = cb; }
    
    DecoderState getState() const { return state; }
    int getCurrentLine() const { return current_line; }
    std::string getCallSign() const { return call_sign; }
    
    // Reset decoder to initial state
    void reset() {
        state = DecoderState::WAITING_FOR_SYNC;
        current_line = 0;
        samples_in_symbol = 0;
        post_sync_samples = 0;
        call_sign.clear();
        
        delete seq0; seq0 = nullptr;
        delete seq1; seq1 = nullptr;
        delete seq2; seq2 = nullptr;
        delete seq3; seq3 = nullptr;
        
        // Reconstruct correlator
        new (&correlator) StreamingSchmidlCox<value, cmplx, search_pos, symbol_len/2, guard_len>(mls0_seq());
        blockdc.samples(2*(symbol_len+guard_len));
    }
    
    // Push a single audio sample (mono, float -1 to 1)
    void pushSample(float sample) {
        // Apply DC block and Hilbert transform for mono
        cmplx tmp = hilbert(blockdc(sample));
        const cmplx* buf = input_hist(tmp);
        
        processSample(buf);
    }
    
private:
    void processSample(const cmplx* buf) {
        switch (state) {
            case DecoderState::WAITING_FOR_SYNC:
                if (correlator(buf)) {
                    symbol_pos = correlator.symbol_pos;
                    cfo_rad = correlator.cfo_rad;
                    
                    if (statusCallback) {
                        std::ostringstream oss;
                        oss << "Sync found! CFO: " << (cfo_rad * rate / (2 * M_PI)) << " Hz";
                        statusCallback(oss.str(), "");
                    }
                    
                    osc.omega(-cfo_rad);
                    
                    // Decode metadata
                    for (int i = 0; i < symbol_len; ++i)
                        tdom[i] = buf[i+symbol_pos+(symbol_len+guard_len)] * osc();
                    fwd(fdom, tdom);
                    
                    CODE::MLS seq4(mls4_poly);
                    for (int i = 0; i < mls4_len; ++i)
                        fdom[bin(i+mls4_off)] *= nrz(seq4());
                    
                    int8_t soft[mls4_len];
                    uint8_t data[(mls4_len+7)/8];
                    for (int i = 0; i < mls4_len; ++i)
                        soft[i] = std::min<value>(std::max<value>(
                            std::nearbyint(127 * (fdom[bin(i+mls4_off)] /
                            fdom[bin(i-1+mls4_off)]).real()), -128), 127);
                    
                    bool unique = osddec(data, soft, genmat);
                    if (!unique) {
                        if (statusCallback) statusCallback("OSD error", "");
                        state = DecoderState::ERROR;
                        return;
                    }
                    
                    uint64_t md = 0;
                    for (int i = 0; i < 55; ++i)
                        md |= (uint64_t)CODE::get_be_bit(data, i) << i;
                    uint16_t cs = 0;
                    for (int i = 0; i < 16; ++i)
                        cs |= (uint16_t)CODE::get_be_bit(data, i+55) << i;
                    
                    crc.reset();
                    if (crc(md<<9) != cs) {
                        if (statusCallback) statusCallback("CRC error", "");
                        state = DecoderState::ERROR;
                        return;
                    }
                    
                    if ((md&255) != 1) {
                        if (statusCallback) statusCallback("Unsupported mode", "");
                        state = DecoderState::ERROR;
                        return;
                    }
                    
                    if ((md>>8) == 0 || (md>>8) >= 129961739795077LL) {
                        if (statusCallback) statusCallback("Invalid call sign", "");
                        state = DecoderState::ERROR;
                        return;
                    }
                    
                    char cs_buf[10];
                    base37_decoder(cs_buf, md>>8, 9);
                    cs_buf[9] = 0;
                    call_sign = cs_buf;
                    
                    if (statusCallback) {
                        statusCallback("Decoding image...", call_sign);
                    }
                    
                    // Initialize MLS sequences for image decoding
                    seq0 = new CODE::MLS(mls0_poly);
                    seq1 = new CODE::MLS(mls1_poly);
                    seq2 = new CODE::MLS(mls2_poly);
                    seq3 = new CODE::MLS(mls3_poly);
                    
                    state = DecoderState::DECODING_IMAGE;
                    current_line = 0;
                    post_sync_samples = 0;
                    samples_in_symbol = 0;
                }
                break;
                
            case DecoderState::DECODING_IMAGE:
                // Count samples and process symbols
                ++post_sync_samples;
                
                // Process image - this is simplified, full impl needs more state
                processImageSample(buf);
                break;
                
            case DecoderState::COMPLETE:
            case DecoderState::ERROR:
                // Do nothing, wait for reset
                break;
        }
    }
    
    void processImageSample(const cmplx* buf) {
        // Calculate expected sample position for each line
        int samples_per_symbol = symbol_len + guard_len;
        int pilot_symbols = (current_line / 8) + 1; // One pilot every 8 lines
        int data_symbols = current_line; // 2 data symbols per 2 lines
        int expected_samples = (symbol_pos + samples_per_symbol) + 
                               pilot_symbols * samples_per_symbol + 
                               data_symbols * samples_per_symbol;
        
        // Check if we have enough samples to decode the next pair of lines
        if (post_sync_samples >= expected_samples + 2 * samples_per_symbol && current_line < img_height) {
            // Decode pilot block if needed
            if (current_line % 8 == 0) {
                int pilot_start = (symbol_pos + samples_per_symbol) + 
                                  (current_line / 8) * samples_per_symbol + 
                                  current_line * samples_per_symbol;
                
                for (int i = 0; i < symbol_len; ++i)
                    tdom[i] = buf[i - post_sync_samples + pilot_start + samples_per_symbol] * osc();
                fwd(fdom, tdom);
                seq1->reset();
                for (int i = 0; i < mls1_len; ++i)
                    chan[i] = nrz((*seq1)()) * fdom[bin(i+mls1_off)];
            }
            
            // Decode two data symbols for two lines
            for (int k = 0; k < 2; ++k) {
                int sym_start = expected_samples + k * samples_per_symbol;
                for (int i = 0; i < symbol_len; ++i)
                    tdom[i] = buf[i - post_sync_samples + sym_start] * osc();
                for (int i = 0; i < guard_len; ++i)
                    osc();
                fwd(fdom+symbol_len*k, tdom);
                
                for (int i = teeth_off, l = 0; i < mls1_len; i += teeth_dist+1, ++l) {
                    fdom[bin(i+mls1_off)+symbol_len*k] *= nrz((*seq0)());
                    index[l] = i+mls1_off;
                    phase[l] = arg(fdom[bin(i+mls1_off)+symbol_len*k] / chan[i]);
                }
                tse.compute(index, phase, teeth_count);
                for (int i = 0; i < mls1_len; ++i)
                    chan[i] *= DSP::polar<value>(1, tse(i+mls1_off));
                for (int i = teeth_off; i < mls1_len; i += teeth_dist+1)
                    chan[i] = fdom[bin(i+mls1_off)+symbol_len*k];
                for (int i = 0, l = 0; i < img_width; ++i, ++l) {
                    if (i % teeth_dist == teeth_off)
                        ++l;
                    fdom[bin(l+mls1_off)+symbol_len*k] /= chan[l];
                    fdom[bin(l+mls1_off)+symbol_len*k] = cmplx(
                        fdom[bin(l+mls1_off)+symbol_len*k].real() * nrz((*seq2)()),
                        fdom[bin(l+mls1_off)+symbol_len*k].imag() * nrz((*seq3)()));
                }
            }
            
            // Convert to RGB
            for (int i = 0, l = 0; i < img_width; i += 2, l += 2) {
                if (i % teeth_dist == teeth_off)
                    ++l;
                cmplx_to_rgb(rgb_line+3*i, rgb_line+3*(i+img_width), fdom+bin(l+mls1_off), fdom+bin(l+mls1_off)+symbol_len);
            }
            
            // Output two lines
            if (lineCallback) {
                // Convert to uint8
                uint8_t rgb_out[2 * img_width * 3];
                for (int i = 0; i < 2 * img_width * 3; ++i) {
                    value v = rgb_line[i] * 255.0f;
                    v = std::max(0.0f, std::min(255.0f, v));
                    rgb_out[i] = (uint8_t)v;
                }
                lineCallback(current_line, 2, rgb_out);
            }
            
            current_line += 2;
            
            if (current_line >= img_height) {
                state = DecoderState::COMPLETE;
                if (statusCallback) {
                    statusCallback("Decode complete!", call_sign);
                }
            }
        }
    }
};
