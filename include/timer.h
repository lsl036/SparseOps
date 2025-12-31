#ifndef TIMER_H
#define TIMER_H
/*
 *  Copyright 2008-2009 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

// A simple timer class

#include"memopt.h"
#include <time.h>
#include"rdtsc.h"
#include"general_config.h"
#include"plat_config.h"
#include<sys/time.h>

class timer
{
    long long int start;
    long long int end;

    public:
    timer()
    { 
        start = rdtsc();
    }

    double milliseconds_elapsed()
    { 
        double elapsed_time;
	    end = rdtsc();
        // elapsed_time = 1000*(end - start)/CPU_FREQUENCY;
        elapsed_time = (double) 1000*(end - start)/CPU_MAX_FREQUENCY;
        return elapsed_time;
    }
    double seconds_elapsed()
    {
        end = rdtsc();
        // return (end - start)/CPU_FREQUENCY;
        return (double)(end - start)/CPU_MAX_FREQUENCY;
    }
};

struct anonymouslib_timer {
    timeval t1, t2;
    struct timezone tzone;

    void start() {
        gettimeofday(&t1, &tzone);
    }
    double stop() {
        gettimeofday(&t2, &tzone);
        double elapsedTime = 0;
        elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
        elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
        return elapsedTime;
    }
};


#endif /* TIMER_H */
