import logging, contextlib, io, time, runpy
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
start = time.time()
stdout_capture = io.StringIO()
with contextlib.redirect_stdout(stdout_capture):
    runpy.run_path('run_tests.py', run_name='__main__')
elapsed = time.time() - start
logging.info('Executed run_tests.py')
logging.info('Execution time: %.2f seconds', elapsed)
output = stdout_capture.getvalue()
logging.info('Stdout:')
logging.info(output)
result = output.count('PASS')
print(result)