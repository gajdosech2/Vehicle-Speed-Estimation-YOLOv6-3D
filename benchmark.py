import os
import re

def compute_avg_fps_in_subfolders():
    results_dir = "/home/photoneo/2016-ITS-BrnoCompSpeed/results/"
    subfolders = [f.path for f in os.scandir(results_dir) if f.is_dir()]
    fps_results = {}

    sort_order = ["nano_b32_480_256_int8", 
                  "nano_b32_640_352_int8",
                  "nano_b32_960_544_int8",
                  "nano_b32_480_256_fp16", 
                  "nano_b32_640_352_fp16",
                  "nano_b32_960_544_fp16",
                  "nano_b32_480_256_fp32", 
                  "nano_b32_640_352_fp32",
                  "nano_b32_960_544_fp32",
                  "nano_distill_b32_480_256_int8", 
                  "nano_distill_b32_640_352_int8",
                  "nano_distill_b32_960_544_int8",
                  "nano_distill_b32_480_256_fp16", 
                  "nano_distill_b32_640_352_fp16",
                  "nano_distill_b32_960_544_fp16",
                  "nano_distill_b32_480_256_fp32", 
                  "nano_distill_b32_640_352_fp32",
                  "nano_distill_b32_960_544_fp32",
                  "small_b32_480_256_int8", 
                  "small_b32_640_352_int8",
                  "small_b32_960_544_int8",
                  "small_b32_480_256_fp16", 
                  "small_b32_640_352_fp16",
                  "small_b32_960_544_fp16",
                  "small_b32_480_256_fp32", 
                  "small_b32_640_352_fp32",
                  "small_b32_960_544_fp32",
                  "small_distill_b32_480_256_int8", 
                  "small_distill_b32_640_352_int8",
                  "small_distill_b32_960_544_int8",
                  "small_distill_b32_480_256_fp16", 
                  "small_distill_b32_640_352_fp16",
                  "small_distill_b32_960_544_fp16",
                  "small_distill_b32_480_256_fp32", 
                  "small_distill_b32_640_352_fp32",
                  "small_distill_b32_960_544_fp32",
                  "medium_b32_480_256_int8", 
                  "medium_b32_640_352_int8",
                  "medium_b32_960_544_int8",
                  "medium_b32_480_256_fp16", 
                  "medium_b32_640_352_fp16",
                  "medium_b32_960_544_fp16",
                  "medium_b32_480_256_fp32", 
                  "medium_b32_640_352_fp32",
                  "medium_b32_960_544_fp32",
                  "large_b32_480_256_int8", 
                  "large_b32_640_352_int8",
                  "large_b32_960_544_int8",
                  "large_b32_480_256_fp16", 
                  "large_b32_640_352_fp16",
                  "large_b32_960_544_fp16",
                  "large_b32_480_256_fp32", 
                  "large_b32_640_352_fp32",
                  "large_b32_960_544_fp32",
                  ]
    
    for subfolder in subfolders:
        txt_files = [f for f in os.listdir(subfolder) if f.endswith('.txt')]
        for txt_file in txt_files:
            file_path = os.path.join(subfolder, txt_file)
            
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        if "Average GPU time:" in line:
                            match = re.search(r"Average GPU time:(\d+)", line)
                            if match:
                                fps = int(match.group(1))
                                if txt_file not in fps_results:
                                    fps_results[txt_file] = []
                                fps_results[txt_file].append((subfolder, fps))
                            break
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
    

    def get_sort_key(config_name):
        for i, substr in enumerate(sort_order):
            if substr in config_name:
                return i
        return len(sort_order) 
    
    sorted_configs = sorted(fps_results.items(), key=lambda x: get_sort_key(x[0]))


    for config_file, data in sorted_configs:
        print(f"Configuration File: {config_file}")
        total_fps = 0
        for subfolder, fps in data:
            print(f"  {subfolder}: {fps} FPS")
            total_fps += fps
        avg_fps = total_fps / len(data) if data else 0
        print(f"  Average FPS: {avg_fps:.2f}")
        print("-" * 40)

if __name__ == "__main__":
    compute_avg_fps_in_subfolders()
