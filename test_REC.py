class memorize(dict):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self[args]

    def __missing__(self, key):
        result = self[key] = self.func(*key)
        return result

def evX(i,o,j):
    #print i,o,j
    if o == "&": return [i[0]*j[0], i[0]*j[1] + i[1]*j[0] + i[1]*j[1]]
    if o == "|": return [i[0]*j[1] + i[1]*j[0], i[0]*j[0] + i[1]*j[1]]
    if o == "^": return [i[0]*j[1] + i[1]*j[0] + i[0]*j[0], i[1]*j[1]]

def ev(exp):
    i,o,j = list(exp)
    i = i=="T"
    j = j=="T"
    if o == "|": return int(i | j)
    if o == "&": return int(i & j)
    if o == "^": return int(i ^ j)

#@memorize
def rec(exp):
    print "GOT", exp
    if len(exp)==1:
        return [int(exp=="T"), int(exp!="T")]
    if len(exp)==3:
        return [ev(exp), 1-ev(exp)] #"T" if ev(exp) else "F"
    if len(exp)>3:
        ts = 0
        fs = 0
        for i in range(1, len(exp), 2):
            print i
            p1,p2 = exp[:i],exp[i+1:]
            print p1,p2
            ti,fi = evX(rec(p1),exp[i],rec(p2))
            #print "tifi", ti, fi
            #print "+", ts
            ts+=ti
            #print "++", ts
            fs+=fi
        print "*",exp,ts,fs
        return ts,fs
def booleanParenthesization(expression):
    return rec(expression)[0]#%1003

rec("T&F^T")