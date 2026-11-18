class ContactDetector:
    """Hysteresis/debounce (PDF 1.3 & 3.5): status berubah hanya jika n sampel berturut-turut berbeda."""
    def __init__(self, n_on=3, n_off=3, threshold=0.5):
        self.n_on, self.n_off, self.thr = n_on, n_off, threshold
        self.state, self.cnt = None, 0

    def update(self, value):
        raw = bool(value > self.thr)
        if self.state is None:
            self.state = raw
            return self.state
        if raw != self.state:
            self.cnt += 1
            if self.cnt >= (self.n_on if raw else self.n_off):
                self.state, self.cnt = raw, 0
        else:
            self.cnt = 0
        return self.state