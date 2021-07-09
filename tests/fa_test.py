from fishauction import FishAuction
from pytest import approx

def test_clone():
    departure_rate=[2,2]
    arrival_rate=[2,2]
    fire_rate=0
    
    
    fa=FishAuction(departure_rate,arrival_rate,fire_rate)
    fa.init_queues(3)
    
    state=fa.step()
    
    fa2=FishAuction.FA_constructor(departure_rate,arrival_rate,fire_rate,state['queues'],state['q_waits'])
    
    assert fa2==fa
    
def test_fire_rate():
    departure_rate=[2,2]
    arrival_rate=[2,2]
    fire_rate=0.1

    fa=FishAuction(departure_rate,arrival_rate,fire_rate)
    
    fa.init_queues(3)
    state=fa.get_state(0)
    counter=0
    its=10000

    for i in range(its):
        state=fa.step(fancy=False)
        if state['fire']: 
            counter+=1

    assert counter/its==approx(fire_rate,0.05)