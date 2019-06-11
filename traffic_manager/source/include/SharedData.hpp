//Declaration of Class member

#pragma once

#include <memory.h>
#include "carla/client/Actor.h"
#include "InMemoryMap.hpp"
#include "SyncQueue.hpp"

namespace traffic_manager{

    class SharedData
    {

    public:
        std::vector<carla::SharedPtr<carla::client::Actor>> registered_actors;
        std::shared_ptr<InMemoryMap> local_map;
        std::map<int , std::shared_ptr<SyncQueue<std::shared_ptr<SimpleWaypoint>>>> buffer_map;
        SharedData();
        ~SharedData();
        void registerActor();
        void deregisterActor();
        
    }; 
}