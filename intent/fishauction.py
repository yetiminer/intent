from collections import deque, namedtuple
import numpy as np
from random import random
from intent.TQ import TillQueue

Chit=namedtuple('Chit',['name','entrytime'])
Receipt=namedtuple('Receipt',['name','entrytime','exittime','status'])

class FishAuction():
    def __init__(self,departure_rate,arrival_rate,fire_rate,q_names=None):
        assert len(departure_rate)==len(arrival_rate)
        self.arrival_rate=arrival_rate
        self.departure_rate=departure_rate
        self.q_num=len(departure_rate)
        if q_names is None: q_names=range(1,self.q_num+1)
        self.queues={q: TillQueue(q,d,a) for q,d,a in zip(q_names,departure_rate,arrival_rate)}
        self.customer_count=1
        self.fire=False
        self.fire_rate=fire_rate
        self.customer_where={}
        self.customer_chits={}
        self.time=0
        self.new_chits=[]
        self.new_receipts=[]
        
    def __repr__(self):
        return str(f'Queues {self.queues}, Records where {self.customer_where}')
        
    def __eq__(self,other):
        #note - makes this unhashable apparently https://stackoverflow.com/questions/1227121/compare-object-instances-for-equality-by-their-attributes
        if not isinstance(other,FishAuction):
            return  NotImplementedError
            
        return all((self.arrival_rate==other.arrival_rate,self.departure_rate==other.departure_rate, 
                self.fire_rate==other.fire_rate, self.queues==other.queues,self.customer_chits==other.customer_chits))
    
    @staticmethod
    def FA_constructor(departure_rate,arrival_rate,fire_rate,queues,queue_times):
        fa=FishAuction(departure_rate,arrival_rate,fire_rate,q_names=queues.keys())
        for q_num,q in queues.items():
            for cust_id,time in zip(q,queue_times[q_num]):
                
                fa.new_arrival(q_num,cust_id,time)
        return fa
        
    @property
    def waiting_status(self):
        return {q_num:q.entry_times_list for q_num,q in self.queues.items()}
    
    @property
    def queue_status(self):
        return {q_num:list(q.q) for q_num,q in self.queues.items()}
        
    
    def get_state(self,cust_id=None):
            state= {'entries':self.new_chits, 'exits':self.new_receipts,
                    'queues':self.queue_status,'q_waits':self.waiting_status,
                    'fire':self.fire}
            if cust_id is not None:
               state['where']=self.where_is(cust_id)
            return state
    @property
    def anon_state(self):
        state= {'entries':{q:len(ch) for q,ch in self.new_chits.items()},
                    'exits':{q:len(rec) for q,rec in self.new_receipts.items()},
                   'queues':{q:de.len for q,de in self.queues.items()},
                    'q_waits':self.waiting_status,
                   'fire':self.fire}
        return state
        
        
    def get_chit(self,cust_id):
        assert cust_id in self.customer_chits
        return customer_chits[cust_id]
        
    def new_customer(self,customer=None,time=None):
        #adds a new customer to the records, gives them a chit and a name if unspecified
        if customer is None: customer=self.customer_count      
        self.customer_count+=1   
        chit=self.record_chit(customer,time)     
        return customer,chit
    
    def decide_fire(self):
        
        fire=False
        if random()<self.fire_rate:
            fire=True
        return fire
        
    
    def step(self,anon=False,fire=None,fancy=True,cust_ID=None):
        self.time+=1
        
        ##decide if market on fire  
        if fire is None: self.fire=self.decide_fire()        
        ##get the new arrivals to the market queues
        self.new_chits=self._process_arrivals()        
        ##get the new departures
        self.new_receipts=self._process_departures(self.fire,fancy)        
        ##format the state information
        
        if anon: state=self.state_anon
        else: state=self.get_state(cust_ID)
        
        return state
        
              
    def _process_departures(self,fire=False,fancy=True):
        #returns the exit receipts and updates the internal records
        if fire:
            #exit without paying
            exit_list={v:q.purge() for v,q in self.queues.items()}            
            status='EWOP'           
        else:
            #exit with paying
            exit_list={v:q.serve() for v,q in self.queues.items()}            
            status='Paid'
        
        rec_dic={q:[self.exit_queue(cust_id,status) for cust_id,_ in zip(*cust_ids)] for q,cust_ids in exit_list.items()}
        
        if fancy: return rec_dic
        
        else: #return in dictionary of numpy arrays, first col=id, second col=entry time
            return {q_num:np.array(deps) for q_num,deps in exit_list.items()},status
    
    def _process_arrivals(self):
        chit_d={}
        
        #get the number of arrivals for each queue
        arrivals=np.random.poisson(self.arrival_rate)
        
        #iterate through each queue and associated arrival to add customers
        for q_num,q_arrivals in zip(range(1,self.q_num+1),arrivals):
                chit_d[q_num]=[self.new_arrival(q_num,cust_id=None) for i in range(q_arrivals)]
        return chit_d 
        
            
    def init_queues(self,r):
        #initiates the queues with even number of customers
        if type(r)==int:
            for v in self.queues:
                for i in range(r):
                    cust_id,chit=self.new_customer()
                    self.enter_queue(v,cust_id)
        elif type(r)==dict:
            for v,chit_list in r.items():
                for chit in chit_list:
                    self.enter_queue(v,)
            
                
                
    def empty_queues(self,status='test'):
        #empties all queues and returns the list of chits
        exit_list=[]
        for v,q in self.queues.items():
            exit_list.append(q.purge())
            
        return [self.exit_queue(cust_id,status) for cust_id,time in exit_list] 
    
    def enter_queue(self,q_num,cust_id,time=None):

        if time is None: time=self.time       
        assert cust_id in self.customer_chits
        try:
            #add the customer to a queue
            self.queues[q_num].add(cust_id,time)
            #record which queue the customer is in
            self.record_where(cust_id,q_num)
        except KeyError:
            print(f'No such Q num {q_num}')
            
    def record_where(self,cust_id,queue):
        assert cust_id not in self.customer_where
        self.customer_where[cust_id]=queue
        
    def record_chit(self,cust_id,time=None):
        if time is None: time=self.time
        assert cust_id not in self.customer_chits
        chit=Chit(cust_id,time)
        self.customer_chits[cust_id]=chit
        return chit
    
    def exit_queue(self,cust_id,status):
        self._record_queue_exit(cust_id)
        assert status in ['EWOP','Paid','test']
        rec=self._exit_chit(cust_id,status)
        return rec
    
    
    def _record_queue_exit(self,cust_id):
        self.customer_where.pop(cust_id)
    
    def _exit_chit(self,cust_id,status):
        assert status in ['EWOP','Paid','test']
        chit=self.customer_chits.pop(cust_id)
        rec= Receipt(cust_id,chit.entrytime,self.time,status)
        return rec
    
    def swap_queue(self,cust_id,new_queue):
        #find the queue
        q_num=self.customer_where[cust_id]
        
        #exit the queue
        self.queues[q_num].exit_queue(cust_id)
        
        #record queue exit 
        self._record_queue_exit(cust_id)
        
        #enter the new queue (and record new position)
        self.enter_queue(new_queue,cust_id)
        
    def new_arrival(self,q_num,cust_id=None,time=None):
        #process a new arrival
        assert q_num in self.queues
        #issue a chit
        cust_id,chit=self.new_customer(cust_id,time)
        #enter a queue (with all the record keeping)
        self.enter_queue(q_num,cust_id,time=time)
        return chit  

    def where_is(self,cust_id):
        #find the queue
        try:
            q_num=self.customer_where[cust_id]
        except KeyError:
            return 0,0
        
        #look for it in the queue
        try:          
            return q_num,self.queues[q_num].where_is(cust_id)
        except IndexError:
            raise
            
class FA_tests():
    def __init__(self,FA):
        self.FA=FA
    
    def _check_records(self):
        for cid in self.FA.customer_where:
            assert cid in self.FA.customer_chits
        for cid in self.FA.customer_chits:
            assert cid in self.FA.customer_where  
        return True
        
    