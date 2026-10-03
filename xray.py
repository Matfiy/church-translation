import sounddevice as sd
import numpy as np

def test_all_channels():
    # We know from your previous list that Dante is Device ID 1
    dev_id = 1 
    dev_info = sd.query_devices(dev_id)
    max_ch = dev_info['max_input_channels']
    
    print(f"Opening X-Ray on {dev_info['name']} ({max_ch} channels)...")

    def callback(indata, frames, time, status):
        # Calculate the maximum volume for every single channel independently
        volumes = np.max(np.abs(indata), axis=0)
        
        active_channels = []
        for i, vol in enumerate(volumes):
            if vol > 0.001:
                # i + 1 because Python counts from 0, but audio engineers count from 1
                active_channels.append((i + 1, vol)) 
                
        if active_channels:
            print(f"🚨 AUDIO FOUND ON CORE AUDIO CHANNEL(S): {active_channels}")
        else:
            print("Scanning 64 channels... pure silence.")

    try:
        # Open a massive 64-channel stream
        with sd.InputStream(device=dev_id, channels=max_ch, samplerate=48000, blocksize=4096, callback=callback):
            print("\nListening! Speak into the stage mic now...")
            print("(Press Ctrl+C to stop the test)\n")
            while True:
                sd.sleep(1000)
    except Exception as e:
        print(f"X-Ray failed to start: {e}")

if __name__ == "__main__":
    test_all_channels()