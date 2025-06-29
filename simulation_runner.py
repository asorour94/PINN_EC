# src/simulation_runner.py

import os
import dynamita.scheduler as ds

class SimulationRunner:
    """
    Encapsulates launching a SUMO simulation job.
    """

    def __init__(self,
                 dll_path: str,
                 init_script: str,
                 variables: list[str],
                 data_cb,
                 msg_cb,
                 parallel_jobs: int = 1,
                 log_level: int = 4):
        # Ensure files exist
        for p in (dll_path, init_script):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"File not found: {p}")

        # Scheduler configuration
        ds.sumo.setParallelJobs(parallel_jobs)
        # Only set log level if provided (None means “use default”)
        if log_level is not None:
            ds.sumo.setLogDetails(log_level)
        ds.sumo.datacomm_callback = data_cb
        ds.sumo.message_callback = msg_cb

        self.dll = dll_path
        self.init = init_script
        self.vars = variables

    def run(self,
            stop_time_ms: int,
            data_comm_ms: int,
            job_data: dict) -> int:
        """
        Launches a dynamic SUMO run.

        Args:
            stop_time_ms: Total run time in milliseconds.
            data_comm_ms: Interval between data callbacks.
            job_data: Dict to accumulate outputs.

        Returns:
            job_id: Identifier for the scheduled run.
        """
        commands = [
            f"execute {self.init}",
            f"set Sumo__StopTime {stop_time_ms}",
            f"set Sumo__DataComm {data_comm_ms}",
            "mode dynamic",
            "start"
        ]

        # Schedule and return job handle
        job_id = ds.sumo.schedule(
            model=self.dll,
            commands=commands,
            variables=self.vars,
            jobData=job_data
        )
        return job_id

    def cleanup(self):
        """Release all scheduler resources."""
        ds.sumo.cleanup()
