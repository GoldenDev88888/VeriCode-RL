import json, urllib.request, re, os

def extract_code(text):
    blocks=re.findall(r'```(?:python|py)?\s*\n(.*?)```',text,re.S|re.I)
    return (blocks[-1] if blocks else text).strip()

class FixtureProvider:
    def __init__(self,name): self.name=name
    def generate(self,task,index):
        # Explicitly scripted outcomes for UI and pipeline tests only.
        return task.reference if self.name=='fixture-reference' or index%2==0 else task.mutants[0]

class OpenAICompatible:
    def __init__(self,model,base_url,key=None,temperature=.7,timeout=90):
        self.name=model; self.url=base_url.rstrip('/')+'/chat/completions'; self.key=key; self.temperature=temperature; self.timeout=timeout
    def generate(self,task,index):
        payload={'model':self.name,'messages':[{'role':'system','content':'Return only Python code defining the requested function. No input/output driver.'},{'role':'user','content':task.prompt}],'temperature':self.temperature,'max_tokens':2048}
        headers={'Content-Type':'application/json'}
        if self.key: headers['Authorization']='Bearer '+self.key
        req=urllib.request.Request(self.url,json.dumps(payload).encode(),headers)
        with urllib.request.urlopen(req,timeout=self.timeout) as r:
            data=json.loads(r.read(2_000_000))
        return extract_code(data['choices'][0]['message']['content'])
