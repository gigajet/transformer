import copy
import operator

class PriorityQueue():
    """
    data: List of 2-tuple (key, value) where key/priority is comparable.
    is_higher_priority: 2-predicate on key/priority where is_higher_priority(x,y) is True iff x is higher priority than y.
        Default is max
    """
    def __init__(self, data = [], is_higher_priority = operator.gt) -> None:
        self.is_higher_priority = is_higher_priority
        self.data = copy.deepcopy(data)
        for i in range(1,len(self)):
            self._upheap(i)

    def _insert(self, elem):
        self.data.append(elem)
        self._upheap(len(self)-1)
        return self

    def insert(self, priority, value):
        return self._insert((priority, value))
    
    """
    Empty the queue
    """
    def clear(self):
        self.data.clear()

    """
    output: the tuple (k,v) with highest priority k
    """
    def pop(self):
        if len(self) == 0:
            raise IndexError("priority queue is empty")
        else:
            k,v=self.data[0]
            self.data[0], self.data[-1] = self.data[-1], self.data[0]
            self.data.pop(-1)
            if len(self) > 0:
                self._downheap(0)
            return (k,v)

    def __len__(self)->int: return len(self.data)


    """
    Impl detail for 0-indexed array
    parent of i is (i-1)//2, beware when i=0
    left child of i is 2i+1
    right child of i is 2i+2
    """
    @staticmethod
    def _p (i:int)->int: return (i-1)//2 if i!=0 else -1
    @staticmethod
    def _lc (i:int)->int: return 2*i+1
    @staticmethod
    def _rc (i:int)->int: return 2*i+2

    def _upheap (self, i: int):
        p = self._p
        while i>0 and self.is_higher_priority(self.data[i][0], self.data[p(i)][0]):
            self.data[i], self.data[p(i)] = self.data[p(i)], self.data[i]
            i=p(i)
    
    def _downheap (self, i: int):
        lc, rc = self._lc, self._rc
        while True:
            j = lc(i)
            if j >= len(self): return
            if rc(i) < len(self) and self.is_higher_priority(self.data[rc(i)][0], self.data[j][0]):
                j = rc(i)
            if self.is_higher_priority(self.data[j][0], self.data[i][0]):
                self.data[i], self.data[j] = self.data[j], self.data[i]
                i = j
            else:
                break

if __name__ == "__main__":
    pq = PriorityQueue([(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)])
    z = []
    while len(pq)>0:
        x,y = pq.pop()
        z.append(y)
    assert(z == [6,5,4,3,2,1])
    pq = PriorityQueue([(1,1),(2,2),(3,3),(4,4),(5,5),(6,6)])
    pq.insert((8,9))
    x,y=pq.pop()
    assert(x==8 and y==9)
    pq.insert((0,-1))
    x,y=pq.pop()
    assert(x==6 and y==6)
    print('Sanity check passed')