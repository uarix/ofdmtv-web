EMCC = em++
EMCC_FLAGS = -std=c++17 -O2 -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=0 \
             -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=134217728 \
             -s EXPORTED_FUNCTIONS='["_malloc","_free"]' \
             -s EXPORTED_RUNTIME_METHODS='["HEAPF32"]' \
             --bind -I../../dsp -I../../code

.PHONY: all clean serve

all: ofdmtv.js

ofdmtv.js: ofdmtv_wasm.cc
	$(EMCC) $(EMCC_FLAGS) -o $@ $<

clean:
	rm -f ofdmtv.js ofdmtv.wasm

serve:
	python3 -m http.server 8080
