Please run main.py to get the autoestimator starting

The scheduler best reward is different from what I obtain from training history
![alt text](https://github.com/YuntianYeAWS/gluon-ts/blob/autogluonts/src/gluonts/autogluonts./scheduler%20result.png?raw=true)

The training curve looks like this

![alt text](https://github.com/YuntianYeAWS/gluon-ts/blob/autogluonts/src/gluonts/autogluonts./training_curve_plot.png?raw=true)

I also have this error come from scheduler.join_jobs() about connection error:


Traceback (most recent call last):
  File "/Users/tokenadmin/anaconda3/lib/python3.7/multiprocessing/managers.py", line 811, in _callmethod
    conn = self._tls.connection
AttributeError: 'ForkAwareLocal' object has no attribute 'connection'
During handling of the above exception, another exception occurred:
Traceback (most recent call last):
  File "/Users/tokenadmin/anaconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py", line 2722, in safe_execfile
    self.compile if shell_futures else None)
  File "/Users/tokenadmin/anaconda3/lib/python3.7/site-packages/IPython/utils/py3compat.py", line 168, in execfile
    exec(compiler(f.read(), fname, 'exec'), glob, loc)
  File "/Users/tokenadmin/Documents/Auto Gluonts/main.py", line 11, in <module>
    a.train()
  File "/Users/tokenadmin/Documents/Auto Gluonts/automodel.py", line 144, in train
    self.scheduler.join_jobs()
  File "/Users/tokenadmin/anaconda3/lib/python3.7/site-packages/autogluon/scheduler/scheduler.py", line 198, in join_jobs
    task_dict['Job'].result(timeout=timeout)
  File "/Users/tokenadmin/anaconda3/lib/python3.7/site-packages/distributed/client.py", line 220, in result
    raise exc.with_traceback(tb)
  File "/Users/tokenadmin/anaconda3/lib/python3.7/site-packages/autogluon/scheduler/scheduler.py", line 169, in _run_dist_job
    ret = return_list[0] if len(return_list) > 0 else None
  File "<string>", line 2, in __len__
  File "/Users/tokenadmin/anaconda3/lib/python3.7/multiprocessing/managers.py", line 815, in _callmethod
    self._connect()
  File "/Users/tokenadmin/anaconda3/lib/python3.7/multiprocessing/managers.py", line 802, in _connect
    conn = self._Client(self._token.address, authkey=self._authkey)
  File "/Users/tokenadmin/anaconda3/lib/python3.7/multiprocessing/connection.py", line 492, in Client
    c = SocketClient(address)
  File "/Users/tokenadmin/anaconda3/lib/python3.7/multiprocessing/connection.py", line 619, in SocketClient
    s.connect(address)
ConnectionRefusedError: [Errno 61] Connection refused
