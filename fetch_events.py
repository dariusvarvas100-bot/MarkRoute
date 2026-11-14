from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess

ROOT=Path(__file__).resolve().parent
EVENTS=(24,135,180,182,283,287,325,346)

def get(event):
    name=ROOT/f'event{event}.json'
    if name.is_file():
        return event,len(json.loads(name.read_text()))
    uri=f'https://api.waterdata.usgs.gov/stnservices/hwms/filteredhwms/?event={event}&format=json'
    try:
        body=subprocess.check_output(['curl','-fLsS','--max-time','35',uri],stderr=subprocess.PIPE)
        data=json.loads(body)
        name.write_bytes(body)
        return event,len(data)
    except (subprocess.CalledProcessError,OSError,ValueError) as exc:
        return event, (exc.stderr.decode(errors='replace').strip() if isinstance(exc,subprocess.CalledProcessError) else str(exc))

def main():
    with ThreadPoolExecutor(max_workers=4) as executor:
        records=list(executor.map(get,EVENTS))
    for record in records:
        print(*record,flush=True)
    if any(not isinstance(count,int) or count==0 for _,count in records):
        raise SystemExit('Some public datasets were unavailable; the bundled panel remains usable.')


if __name__ == '__main__':
    main()
