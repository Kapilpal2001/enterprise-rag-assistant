import psutil

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    ram_mb = mem_info.rss / (1024 * 1024)
    percent = psutil.virtual_memory().percent
    return ram_mb, percent
