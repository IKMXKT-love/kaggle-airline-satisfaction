import concurrent.futures
import json
import urllib.parse
from refresh import ROOT, discovery

REFS=['goodpjw2008/s6e10-goodpjw2008-extra-members-oof','cdeotte/s6e10-tfm-oof-and-test-predictions','najiama/s6e10-oof']

def listing(ref):
    tag=ref.split('/')[0]
    token=None
    files=[]
    receipts=[]
    seen=set()
    complete=False
    for page in range(1,11):
        url='https://www.kaggle.com/api/v1/datasets/list/'+ref
        if token:url+='?'+urllib.parse.urlencode({'pageToken':token})
        r=discovery.fetch(f'{tag}_files_p{page}',url)
        receipts.append(r)
        if 'error' in r:break
        data=json.loads((ROOT/(r['tag']+'.json')).read_text('utf-8'))
        names=[x['name'] for x in data.get('datasetFiles',[])]
        if names and all(n in seen for n in names):
            receipts.append({'error':'Repeated page; pagination not accepted'})
            break
        files.extend(data.get('datasetFiles',[]));seen.update(names)
        token=data.get('nextPageToken')
        if not token:
            complete=True
            break
    result={'ref':ref,'complete':complete,'files':[{'name':x['name'],'bytes':x['totalBytes']} for x in files],'receipts':receipts}
    (ROOT/(tag+'_inventory.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'ref':ref,'complete':complete,'count':len(files),'files':[x['name'] for x in files]},ensure_ascii=False))

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(listing,REFS))
