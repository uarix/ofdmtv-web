/*
OFDM TV WebAssembly Bridge
*/

#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <sstream>

namespace DSP { using std::abs; using std::min; using std::cos; using std::sin; }

#include "complex.hh"
#include "utils.hh"
#include "decibel.hh"
#include "fft.hh"
#include "mls.hh"
#include "crc.hh"
#include "galois_field.hh"
#include "bose_chaudhuri_hocquenghem_encoder.hh"
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

// WAV header
struct WavHeader {
    char riff[4];
    uint32_t fileSize;
    char wave[4];
    char fmt[4];
    uint32_t fmtSize;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
    char data[4];
    uint32_t dataSize;
};

// Memory-based PCM writer for encoding
class MemoryPCM {
    std::vector<float>& buffer;
    int rate_;
    int bits_;
    int channels_;
public:
    MemoryPCM(std::vector<float>& buf, int rate, int bits, int channels) 
        : buffer(buf), rate_(rate), bits_(bits), channels_(channels) {}
    
    int rate() const { return rate_; }
    int bits() const { return bits_; }
    int channels() const { return channels_; }
    
    void write(value *samples, int count, int stride) {
        for (int i = 0; i < count; ++i) {
            if (channels_ == 1) {
                buffer.push_back(samples[i * stride]);
            } else {
                buffer.push_back(samples[i * stride]);
                buffer.push_back(samples[i * stride + 1]);
            }
        }
    }
    
    void silence(int count) {
        for (int i = 0; i < count * channels_; ++i) {
            buffer.push_back(0.0f);
        }
    }
};

// Memory-based image reader for encoding
class MemoryImage {
    const std::vector<uint8_t>& data;
    int width_;
    int height_;
    int pos;
public:
    MemoryImage(const std::vector<uint8_t>& d, int w, int h) 
        : data(d), width_(w), height_(h), pos(0) {}
    
    int width() const { return width_; }
    int height() const { return height_; }
    bool mono() const { return false; }
    
    void read(value *rgb_line, int count) {
        for (int i = 0; i < count * 3 && pos < (int)data.size(); ++i) {
            rgb_line[i] = data[pos++] / 255.0f;
        }
    }
};

// Memory-based image writer for decoding
class MemoryImageWriter {
    std::vector<uint8_t>& data;
public:
    MemoryImageWriter(std::vector<uint8_t>& d) : data(d) {}
    
    void write(value *rgb_line, int count) {
        for (int i = 0; i < count * 3; ++i) {
            value v = rgb_line[i] * 255.0f;
            v = std::max(0.0f, std::min(255.0f, v));
            data.push_back((uint8_t)v);
        }
    }
};

// Streaming image writer with callback for progressive decoding
class StreamingImageWriter {
    std::vector<uint8_t>& data;
    emscripten::val callback;
    int currentLine;
    static const int IMG_WIDTH = 320;
public:
    StreamingImageWriter(std::vector<uint8_t>& d, emscripten::val cb) 
        : data(d), callback(cb), currentLine(0) {}
    
    void write(value *rgb_line, int count) {
        // count is 2 * IMG_WIDTH (2 lines at a time)
        int startIdx = data.size();
        for (int i = 0; i < count * 3; ++i) {
            value v = rgb_line[i] * 255.0f;
            v = std::max(0.0f, std::min(255.0f, v));
            data.push_back((uint8_t)v);
        }
        
        // Call JS callback with new line data
        if (!callback.isNull() && !callback.isUndefined()) {
            int linesWritten = count / IMG_WIDTH;  // Should be 2
            emscripten::val lineData = emscripten::val::global("Uint8Array").new_(count * 3);
            for (int i = 0; i < count * 3; ++i) {
                lineData.set(i, data[startIdx + i]);
            }
            callback(currentLine, linesWritten, lineData);
            currentLine += linesWritten;
        }
    }
};

// Direct PCM reader for decoding - reads from raw pointer
class DirectPCMReader {
    float* buffer;
    unsigned int len;
    int rate_;
    int channels_;
    unsigned int pos;
public:
    DirectPCMReader(float* buf, unsigned int length, int rate, int ch) 
        : buffer(buf), len(length), rate_(rate), channels_(ch), pos(0) {}
    
    int rate() const { return rate_; }
    int channels() const { return channels_; }
    bool good() const { return pos < len; }
    
    // 返回已消耗的样本数
    unsigned int consumed() const { return pos; }
    
    void read(value *samples, int count) {
        for (int i = 0; i < count; ++i) {
            for (int c = 0; c < channels_; ++c) {
                if (pos < len) {
                    samples[i * channels_ + c] = buffer[pos++];
                } else {
                    samples[i * channels_ + c] = 0;
                }
            }
        }
    }
};

// ============= ENCODER =============
template <typename value, typename cmplx, int rate>
struct Encoder
{
    static const int symbol_len = (1280 * rate) / 8000;
    static const int guard_len = symbol_len / 8;
    static const int img_width = 320;
    static const int img_height = 240;
    static const int teeth_count = 16;
    static const int teeth_dist = img_width / teeth_count;
    static const int teeth_off = teeth_dist / 2;
    static const int frame_width = 32;
    static const int mls0_len = 127;
    static const int mls0_poly = 0b10001001;
    static const int mls1_len = img_width + teeth_count;
    static const int mls1_poly = 0b1100110001;
    static const int mls2_poly = 0b10001000000001011;
    static const int mls3_poly = 0b10111010010000001;
    static const int mls4_len = 255;
    static const int mls4_poly = 0b100101011;
    MemoryPCM *pcm;
    DSP::FastFourierTransform<symbol_len, cmplx, 1> bwd;
    CODE::CRC<uint16_t> crc;
    CODE::BoseChaudhuriHocquenghemEncoder<255, 71> bchenc;
    cmplx fdom[2 * symbol_len];
    cmplx tdom[symbol_len];
    cmplx kern0[symbol_len], kern1[symbol_len];
    cmplx guard[guard_len];
    value rgb_line[2 * 3 * img_width];
    cmplx papr_min, papr_max;
    int mls0_off;
    int mls1_off;
    int mls4_off;

    static int bin(int carrier)
    {
        return (carrier + symbol_len) % symbol_len;
    }
    static value nrz(bool bit)
    {
        return 1 - 2 * bit;
    }
    void rgb_to_yuv(value *yuv, const value *rgb)
    {
        value WR(0.299), WB(0.114), WG(1-WR-WB), UMAX(0.493), VMAX(0.877);
        yuv[0] = WR * rgb[0] + WG * rgb[1] + WB * rgb[2];
        yuv[1] = (UMAX/(1-WB)) * (rgb[2]-yuv[0]);
        yuv[2] = (VMAX/(1-WR)) * (rgb[0]-yuv[0]);
    }
    void rgb_to_cmplx(cmplx *out0, cmplx *out1, const value *rgb0, const value *rgb1)
    {
        value yuv0[3], yuv1[3];
        rgb_to_yuv(yuv0, rgb0);
        rgb_to_yuv(yuv1, rgb1);
        out0[0] = cmplx(yuv0[0]+(yuv0[2]+yuv1[2])/2, yuv0[0]-(yuv0[1]+yuv1[1])/2);
        out1[0] = cmplx((yuv0[1]+yuv1[1])/2-yuv1[0], (yuv0[2]+yuv1[2])/2-yuv1[0]);
        rgb_to_yuv(yuv0, rgb0+3);
        rgb_to_yuv(yuv1, rgb1+3);
        out0[1] = cmplx((yuv0[1]+yuv1[1])/2-yuv0[0], (yuv0[2]+yuv1[2])/2-yuv0[0]);
        out1[1] = cmplx(yuv1[0]+(yuv0[2]+yuv1[2])/2, yuv1[0]-(yuv0[1]+yuv1[1])/2);
    }
    void improve_papr(const cmplx *kern)
    {
        for (int n = 0; n < 1000; ++n) {
            int peak = 0;
            for (int i = 1; i < symbol_len; ++i)
                if (norm(tdom[peak]) < norm(tdom[i]))
                    peak = i;
            cmplx orig = tdom[peak];
            for (int i = 0; i < peak; ++i)
                tdom[i] -= orig * kern[symbol_len-peak+i];
            for (int i = peak; i < symbol_len; ++i)
                tdom[i] -= orig * kern[i-peak];
        }
    }
    void symbol(const cmplx *kern = 0)
    {
        bwd(tdom, fdom);
        for (int i = 0; i < symbol_len; ++i)
            tdom[i] /= sqrt(value(8 * symbol_len));
        if (kern)
            improve_papr(kern);
        for (int i = 0; i < guard_len; ++i) {
            value x = value(i) / value(guard_len - 1);
            x = value(0.5) * (value(1) - std::cos(DSP::Const<value>::Pi() * x));
            guard[i] = DSP::lerp(guard[i], tdom[i+symbol_len-guard_len], x);
        }
        cmplx peak, mean;
        for (int i = 0; i < symbol_len; ++i) {
            cmplx power(tdom[i].real() * tdom[i].real(), tdom[i].imag() * tdom[i].imag());
            peak = cmplx(std::max(peak.real(), power.real()), std::max(peak.imag(), power.imag()));
            mean += power;
        }
        if (mean.real() > 0 && mean.imag() > 0) {
            cmplx papr(peak.real() / mean.real(), peak.imag() / mean.imag());
            papr *= value(symbol_len);
            papr_min = cmplx(std::min(papr_min.real(), papr.real()), std::min(papr_min.imag(), papr.imag()));
            papr_max = cmplx(std::max(papr_max.real(), papr.real()), std::max(papr_max.imag(), papr.imag()));
        }
        pcm->write(reinterpret_cast<value *>(guard), guard_len, 2);
        pcm->write(reinterpret_cast<value *>(tdom), symbol_len, 2);
        for (int i = 0; i < guard_len; ++i)
            guard[i] = tdom[i];
    }
    void pilot_block()
    {
        CODE::MLS seq1(mls1_poly);
        value mls1_fac = sqrt(value(symbol_len) / value(mls1_len));
        for (int i = 0; i < symbol_len; ++i)
            fdom[i] = 0;
        for (int i = mls1_off; i < mls1_off + mls1_len; ++i)
            fdom[bin(i)] = mls1_fac * nrz(seq1());
        symbol(kern1);
    }
    void schmidl_cox()
    {
        CODE::MLS seq0(mls0_poly);
        value mls0_fac = sqrt(value(symbol_len) / value(mls0_len));
        for (int i = 0; i < symbol_len; ++i)
            fdom[i] = 0;
        fdom[bin(mls0_off-2)] = mls0_fac;
        for (int i = 0; i < mls0_len; ++i)
            fdom[bin(2*i+mls0_off)] = nrz(seq0());
        for (int i = 0; i < mls0_len; ++i)
            fdom[bin(2*i+mls0_off)] *= fdom[bin(2*(i-1)+mls0_off)];
        symbol();
    }
    void meta_data(uint64_t md)
    {
        uint8_t data[9] = { 0 }, parity[23] = { 0 };
        for (int i = 0; i < 55; ++i)
            CODE::set_be_bit(data, i, (md>>i)&1);
        crc.reset();
        uint16_t cs = crc(md << 9);
        for (int i = 0; i < 16; ++i)
            CODE::set_be_bit(data, i+55, (cs>>i)&1);
        bchenc(data, parity);
        CODE::MLS seq4(mls4_poly);
        value mls4_fac = sqrt(value(symbol_len) / value(mls4_len));
        for (int i = 0; i < symbol_len; ++i)
            fdom[i] = 0;
        fdom[bin(mls4_off-1)] = mls4_fac;
        for (int i = 0; i < 71; ++i)
            fdom[bin(i+mls4_off)] = nrz(CODE::get_be_bit(data, i));
        for (int i = 71; i < mls4_len; ++i)
            fdom[bin(i+mls4_off)] = nrz(CODE::get_be_bit(parity, i-71));
        for (int i = 0; i < mls4_len; ++i)
            fdom[bin(i+mls4_off)] *= fdom[bin(i-1+mls4_off)];
        for (int i = 0; i < mls4_len; ++i)
            fdom[bin(i+mls4_off)] *= nrz(seq4());
        symbol(kern0);
    }
    Encoder(MemoryPCM *pcm, MemoryImage *pel, int freq_off, uint64_t call_sign) :
        pcm(pcm), crc(0xA8F4), bchenc({
            0b100011101, 0b101110111, 0b111110011, 0b101101001,
            0b110111101, 0b111100111, 0b100101011, 0b111010111,
            0b000010011, 0b101100101, 0b110001011, 0b101100011,
            0b100011011, 0b100111111, 0b110001101, 0b100101101,
            0b101011111, 0b111111001, 0b111000011, 0b100111001,
            0b110101001, 0b000011111, 0b110000111, 0b110110001})
    {
        int offset = (freq_off * symbol_len) / rate;
        mls1_off = offset - mls1_len / 2;
        mls0_off = offset - mls0_len + 1;
        mls4_off = offset - mls4_len / 2;
        int car_min = mls1_off - frame_width;
        int car_max = mls1_off+mls1_len + frame_width;
        papr_min = cmplx(1000, 1000), papr_max = cmplx(-1000, -1000);
        for (int i = 0; i < symbol_len; ++i)
            fdom[i] = 0;
        int count = 0;
        for (int i = car_min; i <= car_max; ++i) {
            if (i < mls4_off-1 || i >= mls4_off+mls4_len) {
                fdom[bin(i)] = 1;
                ++count;
            }
        }
        for (int i = 0; i < symbol_len; ++i)
            fdom[i] /= value(10 * count);
        bwd(kern0, fdom);
        for (int i = 0; i < symbol_len; ++i)
            fdom[i] = 0;
        count = 0;
        for (int i = car_min; i <= car_max; ++i) {
            if (i < mls1_off || i >= mls1_off+mls1_len) {
                fdom[bin(i)] = 1;
                ++count;
            }
        }
        for (int i = 0; i < symbol_len; ++i)
            fdom[i] /= value(10 * count);
        bwd(kern1, fdom);
        pilot_block();
        schmidl_cox();
        meta_data((call_sign << 8) | 1);
        CODE::MLS seq0(mls0_poly), seq2(mls2_poly), seq3(mls3_poly);
        value img_fac = sqrt(value(symbol_len) / value(mls1_len));
        for (int i = 0; i < symbol_len; ++i)
            fdom[i] = 0;
        for (int j = 0; j < img_height; j += 2) {
            if (j%8 == 0)
                pilot_block();
            pel->read(rgb_line, 2 * img_width);
            for (int i = 0, l = 0; i < img_width; i += 2, l += 2) {
                if (i % teeth_dist == teeth_off)
                    ++l;
                rgb_to_cmplx(fdom+bin(l+mls1_off), fdom+bin(l+mls1_off)+symbol_len, rgb_line+3*i, rgb_line+3*(img_width+i));
            }
            for (int k = 0; k < 2; ++k) {
                for (int i = 0, l = 0; i < img_width; ++i, ++l) {
                    if (i % teeth_dist == teeth_off) {
                        fdom[bin(l+mls1_off)] = img_fac * nrz(seq0());
                        ++l;
                    }
                    fdom[bin(l+mls1_off)] = img_fac * cmplx(
                        fdom[bin(l+mls1_off)+symbol_len*k].real() * nrz(seq2()),
                        fdom[bin(l+mls1_off)+symbol_len*k].imag() * nrz(seq3()));
                }
                symbol(kern1);
            }
        }
        pilot_block();
        for (int i = 0; i < symbol_len; ++i)
            fdom[i] = 0;
        symbol();
    }
};

// ============= DECODER =============
template <typename value, typename cmplx, int search_pos, int symbol_len, int guard_len>
struct SchmidlCox
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

    SchmidlCox(const cmplx *sequence) : threshold(value(0.17*match_len), value(0.19*match_len))
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
            tmp1[i] = 0;
        for (int i = 0; i < symbol_len; ++i)
            if (norm(tmp0[bin(i-1)]) > 0 &&
                std::min(norm(tmp0[i]), norm(tmp0[bin(i-1)])) * 2 >
                std::max(norm(tmp0[i]), norm(tmp0[bin(i-1)])))
                    tmp1[i] = tmp0[i] / tmp0[bin(i-1)];
        fwd(tmp0, tmp1);
        for (int i = 0; i < symbol_len; ++i)
            tmp0[i] *= kern[i];
        bwd(tmp2, tmp0);

        int shift = 0;
        value peak = 0;
        value next = 0;
        for (int i = 0; i < symbol_len; ++i) {
            value power = norm(tmp2[i]);
            if (power > peak) {
                next = peak;
                peak = power;
                shift = i;
            } else if (power > next) {
                next = power;
            }
        }
        if (peak <= next * 4)
            return false;

        int pos_err = std::nearbyint(arg(tmp2[shift]) * symbol_len / Const::TwoPi());
        if (abs(pos_err) > guard_len / 2)
            return false;
        symbol_pos -= pos_err;

        cfo_rad = shift * (Const::TwoPi() / symbol_len) - frac_cfo;
        if (cfo_rad >= Const::Pi())
            cfo_rad -= Const::TwoPi();
        return true;
    }
};

void base37_decoder(char *str, long long int val, int len)
{
    for (int i = len-1; i >= 0; --i, val /= 37)
        str[i] = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[val%37];
}

template <typename value, typename cmplx, int rate, typename ImageWriter = MemoryImageWriter>
struct Decoder
{
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
    DirectPCMReader *pcm;
    DSP::FastFourierTransform<symbol_len, cmplx, -1> fwd;
    DSP::FastFourierTransform<symbol_len, cmplx, 1> bwd;
    DSP::BlockDC<value, value> blockdc;
    DSP::Hilbert<cmplx, filter_len> hilbert;
    DSP::BipBuffer<cmplx, buffer_len> input_hist;
    DSP::TheilSenEstimator<value, mls1_len> tse;
    SchmidlCox<value, cmplx, search_pos, symbol_len/2, guard_len> correlator;
    CODE::CRC<uint16_t> crc;
    CODE::OrderedStatisticsDecoder<255, 71, 4> osddec;
    int8_t genmat[255*71];
    cmplx chan[mls1_len];
    cmplx fdom[2 * symbol_len], tdom[symbol_len];
    value rgb_line[2 * 3 * img_width];
    value phase[mls1_len], index[mls1_len];
    value cfo_rad, sfo_rad;
    int symbol_pos;
    std::string status;
    std::string call_sign_str;
    bool success;

    static int bin(int carrier)
    {
        return (carrier + symbol_len) % symbol_len;
    }
    static value nrz(bool bit)
    {
        return 1 - 2 * bit;
    }
    void yuv_to_rgb(value *rgb, const value *yuv)
    {
        value WR(0.299), WB(0.114), WG(1-WR-WB), UMAX(0.493), VMAX(0.877);
        rgb[0] = yuv[0] + ((1-WR)/VMAX) * yuv[2];
        rgb[1] = yuv[0] - (WB*(1-WB)/(UMAX*WG)) * yuv[1] - (WR*(1-WR)/(VMAX*WG)) * yuv[2];
        rgb[2] = yuv[0] + ((1-WB)/UMAX) * yuv[1];
    }
    void cmplx_to_rgb(value *rgb0, value *rgb1, cmplx *inp0, cmplx *inp1)
    {
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
    const cmplx *mls0_seq()
    {
        CODE::MLS seq0(mls0_poly);
        for (int i = 0; i < symbol_len/2; ++i)
            fdom[i] = 0;
        for (int i = 0; i < mls0_len; ++i)
            fdom[(i+mls0_off/2+symbol_len/2)%(symbol_len/2)] = nrz(seq0());
        return fdom;
    }
    const cmplx *next_sample()
    {
        cmplx tmp;
        pcm->read(reinterpret_cast<value *>(&tmp), 1);
        if (pcm->channels() == 1)
            tmp = hilbert(blockdc(tmp.real()));
        return input_hist(tmp);
    }
    
    bool isSuccess() const { return success; }
    std::string getStatus() const { return status; }
    std::string getCallSign() const { return call_sign_str; }
    
    Decoder(ImageWriter *pel, DirectPCMReader *pcm, int skip_count) :
        pcm(pcm), correlator(mls0_seq()), crc(0xA8F4), success(false)
    {
        CODE::BoseChaudhuriHocquenghemGenerator<255, 71>::matrix(genmat, true, {
            0b100011101, 0b101110111, 0b111110011, 0b101101001,
            0b110111101, 0b111100111, 0b100101011, 0b111010111,
            0b000010011, 0b101100101, 0b110001011, 0b101100011,
            0b100011011, 0b100111111, 0b110001101, 0b100101101,
            0b101011111, 0b111111001, 0b111000011, 0b100111001,
            0b110101001, 0b000011111, 0b110000111, 0b110110001});

        DSP::Phasor<cmplx> osc;
        blockdc.samples(2*(symbol_len+guard_len));
        const cmplx *buf;
        bool okay;
        do {
            okay = false;
            do {
                if (!pcm->good()) {
                    status = "No signal found";
                    return;
                }
                buf = next_sample();
            } while (!correlator(buf));

            symbol_pos = correlator.symbol_pos;
            cfo_rad = correlator.cfo_rad;
            
            std::ostringstream oss;
            oss << "symbol pos: " << symbol_pos << ", coarse cfo: " << cfo_rad * (rate / Const::TwoPi()) << " Hz";
            status = oss.str();

            osc.omega(-cfo_rad);
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
                status = "OSD error";
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
                status = "CRC error";
                return;
            }
            if ((md&255) != 1) {
                status = "Operation mode unsupported";
                return;
            }
            if ((md>>8) == 0 || (md>>8) >= 129961739795077L) {
                status = "Call sign unsupported";
                return;
            }
            char call_sign[10];
            base37_decoder(call_sign, md>>8, 9);
            call_sign[9] = 0;
            call_sign_str = std::string(call_sign);
            okay = true;
        } while (skip_count--);

        if (!okay)
            return;

        for (int i = 0; i < symbol_pos+(symbol_len+guard_len); ++i)
            buf = next_sample();

        CODE::MLS seq0(mls0_poly), seq1(mls1_poly), seq2(mls2_poly), seq3(mls3_poly);
        for (int j = 0; j < img_height; j += 2) {
            if (j%8==0) {
                for (int i = 0; i < symbol_len; ++i)
                    tdom[i] = buf[i+(symbol_len+guard_len)] * osc();
                fwd(fdom, tdom);
                seq1.reset();
                for (int i = 0; i < mls1_len; ++i)
                    chan[i] = nrz(seq1()) * fdom[bin(i+mls1_off)];
                int count = mls1_len - 1;
                for (int i = 0; i < count; ++i)
                    phase[i] = arg(chan[i+1] / chan[i]);
                std::nth_element(phase, phase+count/2, phase+count);
                value angle_err = phase[count/2];
                int pos_err = std::nearbyint((symbol_len * angle_err) / Const::TwoPi());
                for (int i = 0; i < (symbol_len+guard_len) - pos_err; ++i)
                    buf = next_sample();
                for (int i = 0; i < symbol_len; ++i)
                    tdom[i] = buf[i] * osc();
                for (int i = 0; i < guard_len; ++i)
                    osc();
                fwd(fdom, tdom);
                seq1.reset();
                for (int i = 0; i < mls1_len; ++i)
                    chan[i] = nrz(seq1()) * fdom[bin(i+mls1_off)];
            }
            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < (symbol_len+guard_len); ++i)
                    buf = next_sample();
                for (int i = 0; i < symbol_len; ++i)
                    tdom[i] = buf[i] * osc();
                for (int i = 0; i < guard_len; ++i)
                    osc();
                fwd(fdom+symbol_len*k, tdom);
                for (int i = teeth_off, l = 0; i < mls1_len; i += teeth_dist+1, ++l) {
                    fdom[bin(i+mls1_off)+symbol_len*k] *= nrz(seq0());
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
                        fdom[bin(l+mls1_off)+symbol_len*k].real() * nrz(seq2()),
                        fdom[bin(l+mls1_off)+symbol_len*k].imag() * nrz(seq3()));
                }
            }
            for (int i = 0, l = 0; i < img_width; i += 2, l += 2) {
                if (i % teeth_dist == teeth_off)
                    ++l;
                cmplx_to_rgb(rgb_line+3*i, rgb_line+3*(i+img_width), fdom+bin(l+mls1_off), fdom+bin(l+mls1_off)+symbol_len);
            }
            pel->write(rgb_line, 2 * img_width);
        }
        success = true;
        status = "Decode successful";
    }
};

// Helper function for base37 encoding
long long int base37_encoder(const char *str)
{
    long long int acc = 0;
    for (char c = *str++; c; c = *str++) {
        acc *= 37;
        if (c >= '0' && c <= '9')
            acc += c - '0' + 1;
        else if (c >= 'a' && c <= 'z')
            acc += c - 'a' + 11;
        else if (c >= 'A' && c <= 'Z')
            acc += c - 'A' + 11;
        else if (c != ' ')
            return -1;
    }
    return acc;
}

// ============= WASM API =============

// Encode image to WAV audio
emscripten::val encodeImage(emscripten::val imageData, int width, int height, int sampleRate, std::string callSign) {
    std::vector<uint8_t> imgData;
    unsigned int imgLen = imageData["length"].as<unsigned int>();
    imgData.reserve(width * height * 3);
    
    // Copy RGB data directly (already in RGB format from JS)
    for (unsigned int i = 0; i < imgLen; ++i) {
        imgData.push_back(imageData[i].as<uint8_t>());
    }
    
    std::vector<float> audioBuffer;
    int channels = 1;
    int freqOff = channels == 1 ? 2000 : 0;
    
    long long int callSignVal = base37_encoder(callSign.c_str());
    if (callSignVal <= 0 || callSignVal >= 129961739795077LL) {
        callSignVal = base37_encoder("ANONYMOUS");
    }
    
    MemoryPCM pcm(audioBuffer, sampleRate, 16, channels);
    pcm.silence(sampleRate); // 1 second silence at start
    
    MemoryImage img(imgData, width, height);
    
    switch (sampleRate) {
    case 8000:
        delete new Encoder<value, cmplx, 8000>(&pcm, &img, freqOff, callSignVal);
        break;
    case 16000:
        delete new Encoder<value, cmplx, 16000>(&pcm, &img, freqOff, callSignVal);
        break;
    case 44100:
        delete new Encoder<value, cmplx, 44100>(&pcm, &img, freqOff, callSignVal);
        break;
    case 48000:
        delete new Encoder<value, cmplx, 48000>(&pcm, &img, freqOff, callSignVal);
        break;
    default:
        return emscripten::val::null();
    }
    
    pcm.silence(sampleRate); // 1 second silence at end
    
    // Create WAV file in memory
    int dataSize = audioBuffer.size() * sizeof(int16_t);
    std::vector<uint8_t> wavBuffer(44 + dataSize);
    
    WavHeader header;
    memcpy(header.riff, "RIFF", 4);
    header.fileSize = 36 + dataSize;
    memcpy(header.wave, "WAVE", 4);
    memcpy(header.fmt, "fmt ", 4);
    header.fmtSize = 16;
    header.audioFormat = 1; // PCM
    header.numChannels = channels;
    header.sampleRate = sampleRate;
    header.byteRate = sampleRate * channels * 2;
    header.blockAlign = channels * 2;
    header.bitsPerSample = 16;
    memcpy(header.data, "data", 4);
    header.dataSize = dataSize;
    
    memcpy(wavBuffer.data(), &header, 44);
    
    // Convert float to int16
    int16_t* samples = (int16_t*)(wavBuffer.data() + 44);
    for (size_t i = 0; i < audioBuffer.size(); ++i) {
        float v = audioBuffer[i] * 32767.0f;
        v = std::max(-32768.0f, std::min(32767.0f, v));
        samples[i] = (int16_t)v;
    }
    
    // Return as Uint8Array
    emscripten::val result = emscripten::val::global("Uint8Array").new_(wavBuffer.size());
    for (size_t i = 0; i < wavBuffer.size(); ++i) {
        result.set(i, wavBuffer[i]);
    }
    
    return result;
}

// Decode WAV audio to image - accepts JS Float32Array
emscripten::val decodeAudio(emscripten::val audioData, int sampleRate, int channels) {
    emscripten::val result = emscripten::val::object();
    
    try {
        unsigned int audioLen = audioData["length"].as<unsigned int>();
        
        if (audioLen == 0) {
            result.set("success", false);
            result.set("status", "Empty audio data");
            return result;
        }
        
        // Copy data from JS to C++ vector using typed array view
        std::vector<float> audioBuffer(audioLen);
        emscripten::val memoryView = emscripten::val::module_property("HEAPF32");
        emscripten::val wasmMemory = emscripten::val::module_property("wasmMemory");
        
        // Use the subarray method to copy efficiently
        for (unsigned int i = 0; i < audioLen; ++i) {
            audioBuffer[i] = audioData[i].as<float>();
        }
        
        std::vector<uint8_t> imgData;
        imgData.reserve(320 * 240 * 3);
        
        DirectPCMReader pcm(audioBuffer.data(), audioLen, sampleRate, channels);
        MemoryImageWriter img(imgData);
    
        std::string status;
        std::string callSign;
        bool success = false;
        
        switch (sampleRate) {
        case 8000:
            {
                Decoder<value, cmplx, 8000> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
            }
            break;
        case 16000:
            {
                Decoder<value, cmplx, 16000> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
            }
            break;
        case 44100:
            {
                Decoder<value, cmplx, 44100> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
            }
            break;
        case 48000:
            {
                Decoder<value, cmplx, 48000> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
            }
            break;
        default:
            result.set("success", false);
            result.set("status", "Unsupported sample rate");
            return result;
        }
        
        result.set("success", success);
        result.set("status", status);
        result.set("callSign", callSign);
        
        if (success && imgData.size() >= 320 * 240 * 3) {
            emscripten::val imgArray = emscripten::val::global("Uint8Array").new_((int)imgData.size());
            for (size_t i = 0; i < imgData.size(); ++i) {
                imgArray.set(i, imgData[i]);
            }
            result.set("imageData", imgArray);
            result.set("width", 320);
            result.set("height", 240);
        }
    } catch (const std::exception& e) {
        result.set("success", false);
        result.set("status", std::string("Exception: ") + e.what());
    } catch (...) {
        result.set("success", false);
        result.set("status", "Unknown exception");
    }
    
    return result;
}

// Decode WAV audio to image with streaming/progressive display
emscripten::val decodeAudioStreaming(emscripten::val audioData, int sampleRate, int channels, emscripten::val lineCallback) {
    emscripten::val result = emscripten::val::object();
    
    try {
        unsigned int audioLen = audioData["length"].as<unsigned int>();
        
        if (audioLen == 0) {
            result.set("success", false);
            result.set("status", "Empty audio data");
            return result;
        }
        
        // Copy data from JS to C++ vector
        std::vector<float> audioBuffer(audioLen);
        for (unsigned int i = 0; i < audioLen; ++i) {
            audioBuffer[i] = audioData[i].as<float>();
        }
        
        std::vector<uint8_t> imgData;
        imgData.reserve(320 * 240 * 3);
        
        DirectPCMReader pcm(audioBuffer.data(), audioLen, sampleRate, channels);
        StreamingImageWriter img(imgData, lineCallback);
    
        std::string status;
        std::string callSign;
        bool success = false;
        
        switch (sampleRate) {
        case 8000:
            {
                Decoder<value, cmplx, 8000, StreamingImageWriter> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
            }
            break;
        case 16000:
            {
                Decoder<value, cmplx, 16000, StreamingImageWriter> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
            }
            break;
        case 44100:
            {
                Decoder<value, cmplx, 44100, StreamingImageWriter> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
            }
            break;
        case 48000:
            {
                Decoder<value, cmplx, 48000, StreamingImageWriter> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
            }
            break;
        default:
            result.set("success", false);
            result.set("status", "Unsupported sample rate");
            return result;
        }
        
        result.set("success", success);
        result.set("status", status);
        result.set("callSign", callSign);
        
        if (success && imgData.size() >= 320 * 240 * 3) {
            emscripten::val imgArray = emscripten::val::global("Uint8Array").new_((int)imgData.size());
            for (size_t i = 0; i < imgData.size(); ++i) {
                imgArray.set(i, imgData[i]);
            }
            result.set("imageData", imgArray);
            result.set("width", 320);
            result.set("height", 240);
        }
    } catch (const std::exception& e) {
        result.set("success", false);
        result.set("status", std::string("Exception: ") + e.what());
    } catch (...) {
        result.set("success", false);
        result.set("status", "Unknown exception");
    }
    
    return result;
}

// ============= REALTIME STREAMING DECODER =============
// Global state for realtime decoder
struct RealtimeDecoderState {
    std::vector<float> audioBuffer;      // 环形缓冲区
    int sampleRate;
    bool isActive;
    bool decodeInProgress;               // 正在解码中
    int minBufferSeconds;                // 开始尝试解码的最小秒数
    int maxBufferSeconds;                // 缓冲区最大秒数
    std::string lastCallSign;            // 上次解码的呼号
    emscripten::val statusCallback;
    
    RealtimeDecoderState() : sampleRate(48000), isActive(false), 
        decodeInProgress(false), minBufferSeconds(55), maxBufferSeconds(120) {}
};

static RealtimeDecoderState g_realtimeState;

// Initialize realtime decoder
void initRealtimeDecoder(int sampleRate, emscripten::val lineCb, emscripten::val statusCb) {
    g_realtimeState.audioBuffer.clear();
    g_realtimeState.audioBuffer.reserve(sampleRate * 120); // 预分配120秒
    g_realtimeState.sampleRate = sampleRate;
    g_realtimeState.isActive = true;
    g_realtimeState.decodeInProgress = false;
    g_realtimeState.lastCallSign = "";
    g_realtimeState.statusCallback = statusCb;
    
    if (!statusCb.isNull() && !statusCb.isUndefined()) {
        statusCb(std::string("waiting"), std::string(""));
    }
}

// Push audio samples - 必须尽可能快，不做任何处理
void pushRealtimeSamples(emscripten::val samples) {
    if (!g_realtimeState.isActive) return;
    
    unsigned int len = samples["length"].as<unsigned int>();
    std::vector<float>& buf = g_realtimeState.audioBuffer;
    
    // 直接追加数据
    size_t oldSize = buf.size();
    buf.resize(oldSize + len);
    for (unsigned int i = 0; i < len; ++i) {
        buf[oldSize + i] = samples[i].as<float>();
    }
    
    // 如果缓冲区太大，移除最旧的数据（保持最大120秒）
    int maxSamples = g_realtimeState.sampleRate * g_realtimeState.maxBufferSeconds;
    if ((int)buf.size() > maxSamples) {
        int removeCount = buf.size() - maxSamples;
        buf.erase(buf.begin(), buf.begin() + removeCount);
    }
}

// Try to decode - 只有当缓冲区足够大时才尝试
emscripten::val tryRealtimeDecode() {
    emscripten::val result = emscripten::val::object();
    
    if (!g_realtimeState.isActive) {
        result.set("status", "inactive");
        return result;
    }
    
    if (g_realtimeState.decodeInProgress) {
        result.set("status", "busy");
        return result;
    }
    
    int sampleRate = g_realtimeState.sampleRate;
    std::vector<float>& audioBuffer = g_realtimeState.audioBuffer;
    
    int currentSamples = audioBuffer.size();
    int minSamples = sampleRate * g_realtimeState.minBufferSeconds;
    
    // 缓冲区不够大，继续等待
    if (currentSamples < minSamples) {
        float duration = (float)currentSamples / sampleRate;
        result.set("status", "buffering");
        result.set("duration", duration);
        result.set("needed", g_realtimeState.minBufferSeconds);
        result.set("progress", duration / g_realtimeState.minBufferSeconds * 100);
        return result;
    }
    
    // 缓冲区足够大，尝试解码
    g_realtimeState.decodeInProgress = true;
    
    try {
        std::vector<uint8_t> imgData;
        imgData.reserve(320 * 240 * 3);
        
        class SimpleImageWriter {
            std::vector<uint8_t>& data;
        public:
            SimpleImageWriter(std::vector<uint8_t>& d) : data(d) {}
            
            void write(value *rgb_line, int count) {
                for (int i = 0; i < count * 3; ++i) {
                    value v = rgb_line[i] * 255.0f;
                    v = std::max(0.0f, std::min(255.0f, v));
                    data.push_back((uint8_t)v);
                }
            }
        };
        
        // 使用缓冲区前55秒的数据尝试解码
        int decodeLen = std::min(currentSamples, sampleRate * 55);
        DirectPCMReader pcm(audioBuffer.data(), decodeLen, sampleRate, 1);
        SimpleImageWriter img(imgData);
        
        std::string status;
        std::string callSign;
        bool success = false;
        int samplesConsumed = 0;
        
        switch (sampleRate) {
        case 8000:
            {
                Decoder<value, cmplx, 8000, SimpleImageWriter> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
                samplesConsumed = pcm.consumed();
            }
            break;
        case 16000:
            {
                Decoder<value, cmplx, 16000, SimpleImageWriter> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
                samplesConsumed = pcm.consumed();
            }
            break;
        case 44100:
            {
                Decoder<value, cmplx, 44100, SimpleImageWriter> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
                samplesConsumed = pcm.consumed();
            }
            break;
        case 48000:
            {
                Decoder<value, cmplx, 48000, SimpleImageWriter> dec(&img, &pcm, 0);
                success = dec.isSuccess();
                status = dec.getStatus();
                callSign = dec.getCallSign();
                samplesConsumed = pcm.consumed();
            }
            break;
        }
        
        g_realtimeState.decodeInProgress = false;
        
        int linesDecoded = imgData.size() / (320 * 3);
        
        if (success && linesDecoded >= 240) {
            // 解码成功！
            result.set("status", "success");
            result.set("callSign", callSign);
            result.set("lines", linesDecoded);
            
            // 返回图像数据
            emscripten::val imageData = emscripten::val::global("Uint8Array").new_(imgData.size());
            for (size_t i = 0; i < imgData.size(); ++i) {
                imageData.set(i, imgData[i]);
            }
            result.set("imageData", imageData);
            
            // 移除已解码的数据（使用实际消耗的样本数）
            if (samplesConsumed > 0 && samplesConsumed <= (int)audioBuffer.size()) {
                audioBuffer.erase(audioBuffer.begin(), audioBuffer.begin() + samplesConsumed);
            } else {
                // 如果没有消耗信息，移除约52秒的数据
                int removeCount = std::min((int)audioBuffer.size(), sampleRate * 52);
                audioBuffer.erase(audioBuffer.begin(), audioBuffer.begin() + removeCount);
            }
            
            g_realtimeState.lastCallSign = callSign;
            
            // 回调通知
            if (!g_realtimeState.statusCallback.isNull() && 
                !g_realtimeState.statusCallback.isUndefined()) {
                g_realtimeState.statusCallback(std::string("complete"), callSign);
            }
        } else {
            // 解码失败 - 移除少量数据（约3秒），继续尝试
            // 这样可以找到正确的帧边界
            int removeCount = std::min((int)audioBuffer.size(), sampleRate * 3);
            audioBuffer.erase(audioBuffer.begin(), audioBuffer.begin() + removeCount);
            
            result.set("status", "searching");
            result.set("reason", status);
            result.set("removed", removeCount);
        }
        
    } catch (...) {
        g_realtimeState.decodeInProgress = false;
        
        // 出错时移除少量数据
        int removeCount = std::min((int)audioBuffer.size(), sampleRate * 3);
        audioBuffer.erase(audioBuffer.begin(), audioBuffer.begin() + removeCount);
        
        result.set("status", "error");
    }
    
    return result;
}

// Stop realtime decoder
void stopRealtimeDecoder() {
    g_realtimeState.isActive = false;
    g_realtimeState.decodeInProgress = false;
    g_realtimeState.audioBuffer.clear();
    g_realtimeState.statusCallback = emscripten::val::null();
}

// Reset realtime decoder (for next frame)
void resetRealtimeDecoder() {
    g_realtimeState.audioBuffer.clear();
    g_realtimeState.decodeInProgress = false;
}

// Get buffer status
emscripten::val getRealtimeStatus() {
    emscripten::val result = emscripten::val::object();
    result.set("active", g_realtimeState.isActive);
    result.set("busy", g_realtimeState.decodeInProgress);
    result.set("samples", (int)g_realtimeState.audioBuffer.size());
    result.set("sampleRate", g_realtimeState.sampleRate);
    result.set("duration", (float)g_realtimeState.audioBuffer.size() / g_realtimeState.sampleRate);
    result.set("minNeeded", g_realtimeState.minBufferSeconds);
    result.set("maxBuffer", g_realtimeState.maxBufferSeconds);
    result.set("lastCallSign", g_realtimeState.lastCallSign);
    return result;
}

// Get supported sample rates
emscripten::val getSupportedSampleRates() {
    emscripten::val arr = emscripten::val::array();
    arr.call<void>("push", 8000);
    arr.call<void>("push", 16000);
    arr.call<void>("push", 44100);
    arr.call<void>("push", 48000);
    return arr;
}

EMSCRIPTEN_BINDINGS(ofdmtv_module) {
    emscripten::function("encodeImage", &encodeImage);
    emscripten::function("decodeAudio", &decodeAudio);
    emscripten::function("decodeAudioStreaming", &decodeAudioStreaming);
    emscripten::function("getSupportedSampleRates", &getSupportedSampleRates);
    
    // Realtime decoder functions
    emscripten::function("initRealtimeDecoder", &initRealtimeDecoder);
    emscripten::function("pushRealtimeSamples", &pushRealtimeSamples);
    emscripten::function("tryRealtimeDecode", &tryRealtimeDecode);
    emscripten::function("stopRealtimeDecoder", &stopRealtimeDecoder);
    emscripten::function("resetRealtimeDecoder", &resetRealtimeDecoder);
    emscripten::function("getRealtimeStatus", &getRealtimeStatus);
}
