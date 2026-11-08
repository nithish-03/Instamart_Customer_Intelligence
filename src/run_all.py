from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
for script in ['generate_data.py','src/part_a.py','src/validate.py','src/sql_analysis.py','src/part_b.py','src/part_c.py','src/part_d.py']:
 print(f'\n===== {script} =====',flush=True)
 subprocess.run([sys.executable,str(ROOT/script)],cwd=ROOT,check=True)
