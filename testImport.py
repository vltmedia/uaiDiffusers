# Record Start and End Time

import time

startTime = time.time()


from uaiDiffusers import uaiDiffusers

uaiDiffusers.StartedUp()

endTime = time.time()

print("Time to StartUp: ", endTime - startTime)