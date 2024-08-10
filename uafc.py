import numpy as np
import scipy
import librosa
import sys
import soundfile as sf
import tqdm
import mdct
import pickle
import resampy
import pyppmd
import pickletools
import struct
import argparse
import time
import random
import os

def int_to_bool(int_):
    bytes_arr = b""
    for i in range(len(int_)):
        int_[i] = struct.pack("@B", int(int_[i]))
    for i in int_:
        bytes_arr += bytes(i)
    #print(len(bytes_arr))
    return bytes_arr

def bool_to_int(bool_):
    arr = []
    for i in bool_:
        arr.append(i)
    ##print(arr)
    return arr

def all_same(dat, diff):
    a = 0
    ave = np.mean(np.abs(dat))
    for i in range(len(dat)):
        a += dat[i] - ave
        if np.abs(a) > diff:
            return False
    return True

def set_scale_factor_log(mean, band_num, quality):
    return np.abs(int((((np.sqrt(np.abs(mean)) * (band_num / 4))*2 - quality)) / 4)+1)+1

def func_1(arr):
    if len(arr) == 0:
        return b''
    ###print(arr)
    new_arr = []
    b = []
    a = 0
    for i in arr:
        if np.sign(i) == 0:
            new_arr.append(0)
        elif np.sign(i) == 1:
            new_arr.append(np.abs(i)*2)
        elif np.sign(i) == -1:
            new_arr.append(np.abs(i)*2+1)
        else:
            pass
    ##print(new_arr)
    new_arr = int_to_bool(new_arr)
    ##print(len(new_arr))
    ###print(new_arr)
    return new_arr

def func_2(arr):
    ###print(arr)
    new_arr = []
    a = bool_to_int(arr)
    for i in (a):
        f = int(i / 2)
        if i % 2 == 0:
            new_arr.append(f)
        else:
            new_arr.append(f*-1)
    ###print(new_arr)
    return new_arr

class CHUNK:
    dat1 = b''
    band_num = 0
    mode = 0
    gain = []
    fft_num = 0
    scale_factor = 0
    chanel_type = 0
    pass

class AUDIO:
    dat1 = []
    fs = 48000
    quality = 0
    fft_size = []
    chunk_num = 0
    version = "0.01b"
    ana_band_num = 32
    pass

def flatten_spectrum(signal, window_size=12):
    sign = np.sign(signal)
    padded_signal = np.pad(librosa.db_to_amplitude(signal), (window_size//2, window_size//2), mode='edge')
    smoothed_signal = np.convolve(padded_signal, np.ones(window_size)/window_size, mode='valid')
    return librosa.amplitude_to_db(smoothed_signal)[:len(signal)] * sign

def encode(dat, fs, quality):
    proc_time = time.time()
    print("Quality: " + str(quality-1))
    print("Processing...")
    audio = AUDIO()
    audio.fs = fs
    audio.quality = quality
    playback_time = len(dat[:,0]) / fs
    dat1 = []
    dat2 = []
    mid = (dat[:,0] + dat[:,1]) / 2
    side = (dat[:,0] - dat[:,1]) / 2
    mid = np.power(np.abs(mid)+1e-11, 1.1) * np.sign(mid)
    side = np.power(np.abs(side)+1e-11, 1.2) * np.sign(side)
    mid_ffted = mdct.mdct(mid, framelength=6144, window=scipy.signal.windows.cosine(6144))
    side_ffted = mdct.mdct(side, framelength=6144, window=scipy.signal.windows.cosine(6144))
    mid_phase = np.sign(mid_ffted)
    side_phase = np.sign(side_ffted)
    audio.fft_size = mid_ffted.shape
    mid_db = librosa.amplitude_to_db(mid_ffted, top_db=80)
    side_db = librosa.amplitude_to_db(side_ffted, top_db=64)
    audio.chunk_num = mid_ffted.shape[1]
    for i in tqdm.tqdm(range(mid_ffted.shape[1])):
        db_mid = mid_db[:,i]
        db_side = side_db[:,i]
        phase_mid = mid_phase[:,i]
        phase_side = side_phase[:,i]
        for j in range(audio.ana_band_num-(12-quality)):
            chunk_mid = CHUNK()
            chunk_side = CHUNK()
            chunk_mid.chanel_type = 0
            chunk_side.chanel_type = 1
            chunk_mid.band_num = j
            chunk_side.band_num = j
            chunk_mid.fft_num = i
            chunk_side.fft_num = i
            band_mid = db_mid[j*int(3072/audio.ana_band_num):(j+1)*int(3072/audio.ana_band_num)]
            band_side = db_side[j*int(3072/audio.ana_band_num):(j+1)*int(3072/audio.ana_band_num)]
            band_phs_mid = phase_mid[j*int(3072/audio.ana_band_num):(j+1)*int(3072/audio.ana_band_num)]
            band_phs_side = phase_side[j*int(3072/audio.ana_band_num):(j+1)*int(3072/audio.ana_band_num)]
            #print(band_mid)
            if np.max(band_mid) > 0:
                chunk_mid.mode = 1
                band_mid = librosa.db_to_amplitude(band_mid) * band_phs_mid
            else:
                chunk_mid.mode = 0
            if np.max(band_side) > 0:
                chunk_side.mode = 1
                band_side = librosa.db_to_amplitude(band_side) * band_phs_side
            else:
                chunk_side.mode = 0
            if chunk_mid.mode == 0:
                chunk_mid.gain = np.array([band_mid.min(), band_mid.max()], dtype=np.int8) / 4
                try:
                    if j == 0:
                        chunk_mid.scale_factor = set_scale_factor_log(np.mean(band_mid), chunk_mid.band_num, quality)
                    else:
                        chunk_mid.scale_factor = set_scale_factor_log(np.mean(band_mid), chunk_mid.band_num, quality) + 2
                    if chunk_mid.band_num > 4+quality:
                        if (np.max(np.abs(band_mid)) - np.min(np.abs(band_mid))) < 32-quality*3:
                            band_mid = np.abs(scipy.signal.resample(np.array(band_mid/120, dtype=np.float64), int(len(band_mid)/16)))*120
                            band_phs_mid = np.sign(scipy.signal.resample(np.sign(band_mid), int(len(band_mid)/16)))
                        elif (np.max(np.abs(band_mid)) - np.min(np.abs(band_mid))) < 48-quality*3:
                            band_mid = np.abs(scipy.signal.resample(np.array(band_mid/120, dtype=np.float64), int(len(band_mid)/8)))*120
                            band_phs_mid = np.sign(scipy.signal.resample(np.sign(band_mid), int(len(band_mid)/8)))
                        elif (np.max(np.abs(band_mid)) - np.min(np.abs(band_mid))) < 56-quality*3:
                            band_mid = np.abs(scipy.signal.resample(np.array(band_mid/120, dtype=np.float64), int(len(band_mid)/4)))*120
                            band_phs_mid = np.sign(scipy.signal.resample(np.sign(band_mid), int(len(band_mid)/4)))
                        else:
                            pass
                    chunk_mid.dat1 = np.abs(np.rint(band_mid/chunk_mid.scale_factor)) * band_phs_mid
                except:
                    chunk_mid.dat1 = [0]
                if all_same(chunk_mid.dat1, (10-quality)//1.3):
                    chunk_mid.dat1 = np.array([np.mean(np.abs(chunk_mid.dat1))])
            if chunk_side.mode == 0:
                chunk_side.gain = np.array([band_side.min(), band_side.max()], dtype=np.int8) / 4
                chunk_side.scale_factor = set_scale_factor_log(np.mean(band_side), chunk_side.band_num, quality) + 4
                if chunk_side.band_num > 2+quality:
                    try:
                        if (np.max(np.abs(band_side)) - np.min(np.abs(band_side))) < 32-quality*2:
                            band_side = scipy.signal.resample(np.array(band_side/120, dtype=np.float64), int(len(band_side)/32))*120
                            band_phs_side = np.sign(scipy.signal.resample(np.sign(band_side), int(len(band_mid)/32)))
                        elif (np.max(np.abs(band_side)) - np.min(np.abs(band_side))) < 56-quality*2:
                            band_side = scipy.signal.resample(np.array(band_side/120, dtype=np.float64), int(len(band_side)/16))*120
                            band_phs_side = np.sign(scipy.signal.resample(np.sign(band_side), int(len(band_mid)/16)))
                        elif (np.max(np.abs(band_side)) - np.min(np.abs(band_side))) < 64-quality*3:
                            band_side = scipy.signal.resample(np.array(band_side/120, dtype=np.float64), int(len(band_side)/8))*120
                            band_phs_side = np.sign(scipy.signal.resample(np.sign(band_side), int(len(band_mid)/8)))
                        else:
                            pass
                        chunk_side.dat1 = np.abs(np.rint(band_side/chunk_side.scale_factor)) * band_phs_side
                        if all_same(chunk_side.dat1, (10-quality)):
                            chunk_side.dat1 = np.array([np.mean(np.abs(chunk_side.dat1))])
                    except:
                        chunk_side.dat1 = [0]
            if chunk_mid.mode == 1:
                chunk_mid.scale_factor = (int(np.min(librosa.amplitude_to_db(band_mid))*-1/np.max(np.abs(band_mid))))
                chunk_mid.dat1 = np.rint(band_mid*chunk_mid.scale_factor)
            if chunk_side.mode == 1:
                chunk_side.scale_factor = (int(np.min(librosa.amplitude_to_db(band_side))*-1/np.max(np.abs(band_side))))
                chunk_side.dat1 = np.rint(band_side*chunk_side.scale_factor)
            #print(chunk_mid.dat1)
            #print(chunk_side.dat1)
            chunk_mid.dat1 = func_1(chunk_mid.dat1)
            chunk_side.dat1 = func_1(chunk_side.dat1)
            if chunk_mid.mode > 1:
                chunk_mid.dat1 = False
                chunk_mid.scale_factor = False
            if chunk_side.mode > 1:
                chunk_side.dat1 = False
                chunk_side.scale_factor = False
            #print(chunk_mid.dat1)
            dat1.append(chunk_mid)
            #print(chunk_side.dat1)
            dat1.append(chunk_side)
    audio.dat1 = dat1
    bytes_array = pickle.dumps(audio)
    compressed = pyppmd.compress((bytes_array))
    proc_time = time.time() - proc_time
    print("Processing Time: " + str(int(proc_time)) + "sec")
    print("Bitrate: " + str(int(len(compressed)*8/playback_time/1000*100)/100) + "kbps")
    return compressed

def decode(dat):
    print("Processing...")
    proc_time = time.time()
    audio = pickle.loads(pyppmd.decompress(dat))
    quality = audio.quality
    print("Quality: " + str(quality-1))
    fft_m = np.zeros(audio.fft_size)
    fft_s = np.zeros(audio.fft_size)
    for i in tqdm.tqdm(range(len(audio.dat1))):
        chunk = audio.dat1[i]
        if chunk.mode == 0:
            try:
                if np.abs(chunk.dat1[0]) == 0:
                    chunk.dat1 = np.full(24, -120) + np.random.randint(-6,6,24)
                    chunk.mode = 2
            except:
                chunk.dat1 = np.full(24, -120) + np.random.randint(-6,6,24)
                chunk.mode = 2
            chunk.dat1 = func_2(chunk.dat1)
            if len(chunk.dat1) == 1:
                if chunk.chanel_type == 0:
                    chunk.dat1 = np.full(int(3072/audio.ana_band_num),chunk.dat1[0]) * ([random.choice([1, -1]) for x in range(int(3072/audio.ana_band_num))]) + np.int32(np.random.randn(len(chunk.dat1)))
                if chunk.chanel_type == 1:
                    chunk.dat1 = np.full(int(3072/audio.ana_band_num),chunk.dat1[0]) * ([random.choice([1, -1]) for x in range(int(3072/audio.ana_band_num))]) + np.int32(np.random.randn(len(chunk.dat1)))
                phs = ([1 if np.random.randn() < 0 else -1 for x in range(int(3072/audio.ana_band_num))])
                chunk.dat1 *= phs
            if len(chunk.dat1) != int(3072/audio.ana_band_num):
                phs = ([1 if np.random.randn() < 0 else -1 for x in range(int(3072/audio.ana_band_num))])
                phs_2 = np.sign(chunk.dat1)
                for i in range(len(chunk.dat1)):
                    phs[i*int((int(3072/audio.ana_band_num))/len(chunk.dat1))] = phs_2[i]
                chunk.dat1 = scipy.signal.resample(chunk.dat1, int((int(3072/audio.ana_band_num))))
                chunk.dat1 *= np.int32(phs)
            #print(len(chunk.dat1))
            chunk.dat1 = np.array(chunk.dat1)
            chunk.dat1 *= chunk.scale_factor
            #chunk.dat1 += np.int32(np.random.random(len(chunk.dat1)) * 4)
            chunk.dat1 = np.abs(np.clip(chunk.gain[0]*4, chunk.gain[1]*4, np.abs(chunk.dat1)*-1)) * np.sign(chunk.dat1)
            if chunk.chanel_type == 0:
                fft_m[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)] = librosa.db_to_amplitude(-1*np.abs(chunk.dat1)) * np.sign(chunk.dat1)
            if chunk.chanel_type == 1:
                fft_s[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)] = librosa.db_to_amplitude(-1*np.abs(chunk.dat1)) * np.sign(chunk.dat1)
        if chunk.mode == 1:
            chunk.dat1 = func_2(chunk.dat1)
            chunk.dat1 = np.array(chunk.dat1)
            if chunk.chanel_type == 0:
                fft_m[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)] = chunk.dat1 / chunk.scale_factor
            if chunk.chanel_type == 1:
                fft_s[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)] = chunk.dat1 / chunk.scale_factor
        if chunk.mode == 2:
            if chunk.chanel_type == 0:
                if np.max(np.abs(fft_s[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)])) != 0:
                    fft_m[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)] += np.abs(fft_s[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)] / 2) * ([random.choice([1, -1]) for x in range(int(3072/audio.ana_band_num))][::-1])
                if chunk.band_num > 15 and chunk.band_num < 27:
                    rand = random.randint(1,6)
                    fft_m[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)] += librosa.db_to_amplitude(librosa.amplitude_to_db((np.abs(fft_m[:,chunk.fft_num][(chunk.band_num-rand)*int(3072/audio.ana_band_num):(chunk.band_num-rand+1)*int(3072/audio.ana_band_num)] / 1.88) * ([random.choice([1, -1]) for x in range(int(3072/audio.ana_band_num))]))[::-1]))
                pass
            if chunk.chanel_type == 1:
                if np.max(librosa.amplitude_to_db(np.abs(fft_m[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)]))) < -36:
                    fft_s[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)] += librosa.db_to_amplitude(librosa.amplitude_to_db(np.abs(fft_m[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)]))-12) * ([random.choice([1, -1]) for x in range(int(3072/audio.ana_band_num))])
                elif chunk.band_num > 1 and chunk.band_num < 27:
                    fft_s[:,chunk.fft_num][chunk.band_num*int(3072/audio.ana_band_num):(chunk.band_num+1)*int(3072/audio.ana_band_num)] += (np.median(np.abs(fft_m[:,chunk.fft_num][(chunk.band_num-1)*int(3072/audio.ana_band_num):(chunk.band_num)*int(3072/audio.ana_band_num)], dtype=np.float64)) / 4 * np.array([random.choice([1.0, -1.0]) for x in range(int(3072/audio.ana_band_num))], dtype=np.float64))[::-1]
        else:
            pass
    fft_m = np.array(fft_m)
    fft_s = np.array(fft_s)
    mid = mdct.imdct(fft_m, framelength=6144)
    side = mdct.imdct(fft_s, framelength=6144)
    mid = (np.abs(mid)**1.1) * np.sign(mid)
    side = (np.abs(side)**1.2) * np.sign(side)
    proc_time = np.abs(proc_time - time.time())
    print("Processing Time: " + str(int(proc_time)) + "sec")
    #print(side.tolist())
    return np.array([(mid+side), (mid-side)]).T, audio.fs

def main():
    parser = argparse.ArgumentParser(description="Audio encoding and decoding tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Encode parser
    encode_parser = subparsers.add_parser("encode", help="Encode audio file")
    encode_parser.add_argument("input", help="Input audio file")
    encode_parser.add_argument("output", help="Output .ua2 file")
    encode_parser.add_argument("quality", type=float, help="Encoding quality (0-9.99)")

    # Decode parser
    decode_parser = subparsers.add_parser("decode", help="Decode .ua2 file")
    decode_parser.add_argument("input", help="Input .ua2 file")
    decode_parser.add_argument("output", help="Output audio file")

    # Test parser
    test_parser = subparsers.add_parser("test", help="Run test on audio file")
    test_parser.add_argument("input", help="Input audio file")
    test_parser.add_argument("quality", type=int, help="Test quality (0-9.99)")

    args = parser.parse_args()

    if args.command == "encode":
        dat, fs = sf.read(args.input)
        quality = int(args.quality) + 1
        if quality >= 10:
            print("quality must be 0~9.99")
            return 0
        dat = encode(dat, fs, quality)
        with open(args.output, "wb") as f:
            f.write(dat)

    elif args.command == "decode":
        with open(args.input, "rb") as f:
            dat = f.read()
        dat, fs = decode(dat)
        sf.write(args.output, dat, fs, format="WAV")

    elif args.command == "test":
        dat, fs = sf.read(args.input)
        quality = args.quality + 1
        test(dat, quality)

    print("Finish!")
    return 0

if __name__ == '__main__':
    main()
