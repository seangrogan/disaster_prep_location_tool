import time
import warnings


class StoppingCriteria:
    def __init__(self, pars):
        self.global_max_time = pars.get("global_time_limit", None)
        self.global_max_iter_count = pars.get("global_iter_count", None)
        self.local_max_time = pars.get("local_max_time", None)
        self.local_max_iter_count = pars.get("local_iter_count", None)

        self.global_start_time = None
        self.local_start_time = None
        self.global_iter_count = 0
        self.local_iter_count = 0
        if not any([
            self.global_max_time,
            self.global_max_iter_count,
            self.local_max_time,
            self.local_max_iter_count
        ]):
            self.global_max_iter_count = 1000
            self.local_max_iter_count = 100
            warnings.warn(f"No stopping criteria passed.  Using a max global iter "
                          f"count of {self.global_max_iter_count} and "
                          f"local iter count of {self.local_max_iter_count}")

    def start(self, allow_warn=True):
        if self.local_start_time:
            warnings.warn(f"Warning! Calling .start() a second time!  This resets the global timer!")
            if not allow_warn:
                raise Exception("You can override this exception by setting allow_warn=True")
        self.global_start_time = time.time()
        self.local_start_time = time.time()

    def reset_local_timer(self):
        self.local_start_time = time.time()

    def reset_local_counter(self):
        self.local_iter_count = 0

    def update_counter(self, glbl_iter=1, local_iter=1):
        self.global_iter_count += glbl_iter
        self.local_iter_count += local_iter

    def update_local_counter(self, local_iter=1):
        self.local_iter_count += local_iter

    def update_global_counter(self, glbl_iter=1):
        self.global_iter_count += glbl_iter

    def is_not_complete(self):
        return not self.is_complete()

    def is_complete(self):
        if self.global_max_iter_count and self.global_iter_count > self.global_max_iter_count:
            return True
        if self.local_max_iter_count and self.local_iter_count > self.local_max_iter_count:
            return True
        if self.local_max_time and self.local_max_time > time.time() - self.local_start_time:
            return True
        if self.global_max_time \
                and self.global_max_time \
                and self.global_max_time > time.time() - self.global_start_time:
            return True
        return False