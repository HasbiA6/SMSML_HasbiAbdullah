import time
import psutil
from prometheus_client import Gauge, start_http_server

SYSTEM_CPU_PERCENT = Gauge("system_cpu_percent", "System CPU usage percent")
SYSTEM_MEM_PERCENT = Gauge("system_memory_percent", "System memory usage percent")
SYSTEM_MEM_AVAILABLE = Gauge("system_memory_available_bytes", "Available memory in bytes")
SYSTEM_DISK_USAGE = Gauge("system_disk_usage_percent", "Disk usage percent")
SYSTEM_DISK_FREE = Gauge("system_disk_free_bytes", "Disk free bytes")
EXPORTER_UPTIME = Gauge("exporter_uptime_seconds", "Exporter uptime seconds")

START_TIME = time.time()

def collect_loop(interval: float = 2.0):
    while True:
        SYSTEM_CPU_PERCENT.set(psutil.cpu_percent(interval=None))

        vm = psutil.virtual_memory()
        SYSTEM_MEM_PERCENT.set(vm.percent)
        SYSTEM_MEM_AVAILABLE.set(vm.available)

        du = psutil.disk_usage("/")
        SYSTEM_DISK_USAGE.set(du.percent)
        SYSTEM_DISK_FREE.set(du.free)

        EXPORTER_UPTIME.set(time.time() - START_TIME)
        time.sleep(interval)

if __name__ == "__main__":
    # expose /metrics on :9000
    start_http_server(9000)
    collect_loop()