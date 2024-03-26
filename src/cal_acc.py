


class MyAverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n

    def get_average(self):
        self.avg = self.sum / (self.count + 1e-12)
        return self.avg
    
class GetTarget:
    def __init__(self):
        self.TP = 0 
        self.FP = 0 
        self.TN = 0 
        self.FN = 0  
    def insert(self, pair):
        true = pair[0]
        pred = pair[1]
        if(true == 1 and pred == 1):
            self.TP += 1
        elif(true == 0 and pred == 1):
            self.FP += 1
        elif(true == 1 and pred == 0):
            self.FN += 1
        elif(true == 0 and pred == 0):
            self.TN += 1
    def getRecall(self):
        if(self.TP+self.FN == 0): return 0
        res = self.TP/(self.TP+self.FN)
        return res
    def getPrec(self):
        if(self.TP+self.FP == 0): return 0
        res = self.TP/(self.TP+self.FP)
        return res
    def getSEN(self):
        if(self.TN+self.FP == 0): return 0
        res = self.TN / (self.TN+self.FP)
        return res
    def getF1(self):
        if(self.getPrec()+self.getRecall() == 0):return 0
        res = 2*(self.getPrec()*self.getRecall())/(self.getPrec()+self.getRecall())
        return res