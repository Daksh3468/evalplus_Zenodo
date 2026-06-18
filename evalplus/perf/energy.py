"""
This module aims at providing energy consumption measurements
"""
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from io import StringIO
from typing import Any, Optional

import polars as pl
import rich
from polars.polars import NoDataError

PERF_COEFFICIENT: Optional[
    float] = None  # If set, the coefficient to multiple the duration counter in perf output in order to obtain the duration in seconds. Varies between hardwares


@dataclass
class EnergyMeasurement:
    cpu_energy: float = float()
    gpu_energy: float = float()
    total_energy: float = field(init=False)
    duration: float = float()  # In seconds
    perf_duration: float = float()  # Duration measured by the perf process

    def __post_init__(self):
        self.total_energy = self.cpu_energy + self.gpu_energy

    def __add__(self, other):
        return EnergyMeasurement(cpu_energy=self.cpu_energy + other.cpu_energy,
                                 gpu_energy=self.gpu_energy + other.gpu_energy,
                                 duration=self.duration + other.duration,
                                 perf_duration=self.perf_duration + other.perf_duration)


def measure_energy(gpu=True):
    def _measure_energy(func):
        def wrapper(*args, **kwargs):
            return monitor_energy_for_function(gpu, func, *args, **kwargs)

        return wrapper

    return _measure_energy


def monitor_energy_for_function(gpu, func, *args, **kwargs) -> tuple[Any, EnergyMeasurement]:
    global PERF_COEFFICIENT
    assert PERF_COEFFICIENT is not None, "PERF_COEFFICIENT must be set, did you forget to calibrate?"
    monitor = EnergyMonitor(gpu)
    # monitor.check_measuring_is_possible()
    monitor.start_measure()
    result = func(*args, **kwargs)
    energy = monitor.end_measures()
    rich.print(f"[bold green]Energy consumed: {energy}[/]")
    return result, energy


def calibrate(time_s: float, gpu=True) -> EnergyMeasurement:
    """Sleeps for time_s seconds and measures energy consumption"""
    global PERF_COEFFICIENT
    should_calibrate_perf_coefficient = not PERF_COEFFICIENT
    if should_calibrate_perf_coefficient:
        # Temporary perf coefficient
        PERF_COEFFICIENT = 1 / 10 ** 9

    monitor = EnergyMonitor(gpu)
    monitor.check_measuring_is_possible()
    if should_calibrate_perf_coefficient:
        print("Perf coefficient not set, calibrating")
        monitor._start_perf_calibration(
            time_s / 2)  # We divide by two so that the timeout has time to be done in the calibration
    monitor.start_measure()
    time.sleep(time_s)
    if should_calibrate_perf_coefficient:
        monitor._end_perf_calibration(time_s / 2)
        print(f"Perf coefficient set to {PERF_COEFFICIENT}")
    energy_measured = monitor.end_measures()
    rich.print(f"[bold green]Energy consumed: {energy_measured}[/]")
    return energy_measured


class EnergyMonitor(object):
    def __init__(self, measure_gpu):
        self.measure_gpu = measure_gpu

        self._cpu_energy = 0
        self._perf_process = None
        self._perf_calibration_process = None
        self._nvidia_smi_process = None
        self._gpu_measure_start_time = None
        self._gpu_measure_end_time = None
        self._start_time = None
        self._end_time = None

    def check_measuring_is_possible(self):
        self.start_measure()
        time.sleep(0.2)
        self.end_measures()
        # Everything is okay if no exception happened.
        return

    def start_measure(self):
        self._start_cpu_measure()
        if self.measure_gpu:
            self._start_gpu_measure()
        self._start_time = time.time()

    def end_measures(self, filter_gpus: list[int] | None = None):
        self._end_time = time.time()
        cpu_energy, perf_duration = self._end_cpu_measure()
        if self.measure_gpu:
            gpu_energy = self._end_gpu_measure(filter_gpus)
        else:
            gpu_energy = 0
        return EnergyMeasurement(round(cpu_energy, 2), round(gpu_energy, 2), duration=self._end_time - self._start_time,
                                 perf_duration=perf_duration)

    def _start_cpu_measure(self):
        if self._perf_process:
            self._perf_process.kill()
        self._perf_process = subprocess.Popen("perf stat -e power/energy-pkg/ -x\\;", shell=True,
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    def _start_gpu_measure(self):
        if self._nvidia_smi_process:
            self._nvidia_smi_process.kill()
        self._nvidia_smi_process = subprocess.Popen(
            "nvidia-smi --query-gpu=index,timestamp,power.draw --format=csv,nounits -lms 1000", shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
        self._gpu_measure_start_time = time.time()

    def _start_perf_calibration(self, time_s: float):
        self._perf_calibration_process = subprocess.Popen(
            f"perf stat -e power/energy-pkg/ -x\\; --timeout {round(time_s * 10 ** 3)}", shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)

    def _end_perf_calibration(self, time_s: float):
        stderr = self._kill_perf(self._perf_calibration_process)
        duration_counter = float(stderr.split(";")[3])
        global PERF_COEFFICIENT
        PERF_COEFFICIENT = time_s / duration_counter

    def _end_cpu_measure(self):
        stderr = self._kill_perf(self._perf_process)
        try:
            energy_total = float(stderr.split(";")[0])  # energy consumption in Joules
            perf_duration = float(stderr.split(";")[3]) * PERF_COEFFICIENT  # From a counter to seconds
        except ValueError as e:
            rich.print(f"[bold red]Error while converting CPU energy measures, using default value of 0: {e}[/]")
            energy_total = 0
            perf_duration = 0
        return energy_total, perf_duration

    def _kill_perf(self, process):
        os.killpg(os.getpgid(process.pid), signal.SIGINT)
        try:
            process.wait(5)
        except subprocess.TimeoutExpired as e:
            rich.print(f"[red]The perf command had a timeout. Its output may not be reliable? : {e}[/red]")
        stderr = process.communicate()[1].decode("ascii").replace(",", ".")
        if "<not supported>" in stderr:
            print("It seems your system does not support instruction counting.")
            print("Try this on Linux:")
            print('echo "-1" | sudo tee -a /proc/sys/kernel/perf_event_paranoid')
            raise Exception("Instruction counting not supported")
        return stderr

    def _end_gpu_measure(self, filter_gpus):
        os.killpg(os.getpgid(self._nvidia_smi_process.pid), signal.SIGINT)
        self._gpu_measure_end_time = time.time()
        try:
            self._nvidia_smi_process.wait(5)
        except subprocess.TimeoutExpired as e:
            rich.print(f"[red]The nvidia-smi command had a timeout. Its output may not be reliable? : {e}[/red]")
        gpu_csv = self._nvidia_smi_process.communicate()[0].decode("ascii").replace(", ", ",")
        stderr = self._nvidia_smi_process.communicate()[1].decode("ascii")
        if "nvidia-smi: not found" in stderr:
            print(
                "There is no nvidia-smi installed on your system. Please install it to measure GPU energy consumption.")
            print(stderr)
            raise Exception("nvidia-smi not found")
        try:
            # Aggregate all the data from the csv
            with StringIO(gpu_csv) as f:
                gpu_df: pl.DataFrame = pl.read_csv(f, separator=",")
            if filter_gpus is not None and len(filter_gpus) > 0:
                gpu_df = gpu_df.filter(pl.col("index").is_in(filter_gpus))
            powerdraw_column = "power.draw [W]"
            gpu_df = gpu_df.cast({powerdraw_column: pl.Float32})
            average_powerdraw = gpu_df.group_by("index").agg([pl.col(powerdraw_column).mean()])[powerdraw_column].sum()

            measure_duration_s = self._gpu_measure_end_time - self._gpu_measure_start_time
            total_powerdraw_j = average_powerdraw * measure_duration_s
            return total_powerdraw_j
        except NoDataError as e:
            rich.print(f"[red]Error while converting GPU measures, using default value of 0: {e}[/]")
            return 0


if __name__ == "__main__":
    monitor = EnergyMonitor()
    monitor.check_measuring_is_possible()
    monitor.start_measure()
    time.sleep(5)
    energy = monitor.end_measures()
    print(energy)
