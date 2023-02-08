# class _RefGet:
#     def __set_name__(self, owner, name):
#         self.name = name

#     def __get__(self, ref, owner):
#         return getattr(ref.shared, self.name)[ref.idx :]


# @dataclass
# class Ref:
#     shared: "SharedPeriods"
#     idx: int

#     data = _RefGet()
#     year = _RefGet()
#     month = _RefGet()
#     day = _RefGet()
#     hour = _RefGet()
#     minute = _RefGet()
#     second = _RefGet()
#     dayofweek = _RefGet()
#     dayofyear = _RefGet()
#     week = _RefGet()


# class _ViewGet:
#     def __set_name__(self, owner, name):
#         self.name = name

#     def __get__(self, view, owner):
#         return getattr(view.ref, self.name)[: view.length]


# @dataclass
# class View:
#     ref: ref
#     length: int

#     data = _ViewGet()
#     year = _ViewGet()
#     month = _ViewGet()
#     day = _ViewGet()
#     hour = _ViewGet()
#     minute = _ViewGet()
#     second = _ViewGet()
#     dayofweek = _ViewGet()
#     dayofyear = _ViewGet()
#     week = _ViewGet()

#     def future(self, count: int) -> Periods:
#         return self.ref.shared.get(self.data[-1] + 1, count)

#     def past(self, count: int) -> Periods:
#         return self.ref.shared.get(self.data[0] - count, count)

#     def extend(self, count: int) -> Periods:
#         return self.ref.shared.get(self.data[0], self.length + count)

#     def __len__(self):
#         return len(self.length)


# @dataclass(eq=False)
# class SharedPeriods:
#     freq: tuple[str, int]
#     data: Optional[np.ndarray] = None
#     refs: dict = field(default_factory=dict)

#     year: np.ndarray = field(init=False)
#     month: np.ndarray = field(init=False)
#     day: np.ndarray = field(init=False)
#     hour: np.ndarray = field(init=False)
#     minute: np.ndarray = field(init=False)
#     second: np.ndarray = field(init=False)
#     dayofweek: np.ndarray = field(init=False)
#     dayofyear: np.ndarray = field(init=False)
#     week: np.ndarray = field(init=False)

#     def _generate_features(self):
#         periods = Periods(self.data, self.freq[1])
#         self.year = periods.year
#         self.month = periods.month
#         self.day = periods.day
#         self.hour = periods.hour
#         self.minute = periods.minute
#         self.second = periods.second
#         self.dayofweek = periods.dayofweek
#         self.dayofyear = periods.dayofyear
#         self.week = periods.week

#     def _ref_for(self, idx):
#         if idx not in self.refs:
#             self.refs[idx] = Ref(self, idx)

#         ref = self.refs[idx]
#         return ref

#     def _shift_refs(self, count):
#         for ref in self.refs.values():
#             ref.idx += count

#         self.refs = {ref.idx: ref for ref in self.refs.values()}

#     def get(self, start, count):
#         freq, n = self.freq
#         start = np.datetime64(start, freq)
#         last = start + n * count

#         if self.data is None:
#             self.data = np.arange(start, n * count, n)
#             self._generate_features()
#             idx = 0
#         else:
#             offset = (start - self.data[0]).astype(int)
#             if offset < 0:
#                 new_last = max(self.data[-1], last)
#                 self.data = np.arange(start, new_last + 1, n)
#                 self._generate_features()
#                 self._shift_refs(-offset)
#                 idx = 0
#             elif offset >= len(self.data) or last > self.data[-1]:
#                 # new start or last, is behind our last, regenerate
#                 self.data = np.arange(self.data[0], last + 1, n)
#                 self._generate_features()
#                 idx = offset
#             else:
#                 idx = offset

#         return View(self._ref_for(idx), count)


# periods_cache = {}


# def periods(data, freq, count: int):
#     freq_str, multiple = freq = normalize_freq(freq)
#     start = np.datetime64(data, freq_str)

#     if freq not in periods_cache:
#         periods_cache[freq] = SharedPeriods(freq)

#     return periods_cache[freq].get(start, count)
