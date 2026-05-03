import threading


class ExperimentControlState:
    def __init__(self):
        self._lock = threading.Lock()
        self._paused = False
        self._reset_requested = False
        self._requested_group_index = None
        self._requested_trial_index = None

    def pause(self):
        with self._lock:
            self._paused = True

    def resume(self):
        with self._lock:
            self._paused = False

    def toggle_pause(self):
        with self._lock:
            self._paused = not self._paused
            return self._paused

    def is_paused(self):
        with self._lock:
            return self._paused

    def request_reset(self, group_index=None, trial_index=None):
        with self._lock:
            self._reset_requested = True
            self._requested_group_index = group_index
            self._requested_trial_index = trial_index

    def consume_reset_request(self):
        with self._lock:
            if not self._reset_requested:
                return False, None, None
            self._reset_requested = False
            group_index = self._requested_group_index
            trial_index = self._requested_trial_index
            self._requested_group_index = None
            self._requested_trial_index = None
            return True, group_index, trial_index
