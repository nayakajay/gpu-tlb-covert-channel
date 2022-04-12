# Example outputs
Here we list the example outputs which we observed during our evaluation.
Note that value > THRESHOLD (i.e., 420 here). is a 0, else the value transmitted is a 1.
For example, a row with content ```k: 8 latency, 437.000 clk``` means received a bit ```0``` as the 8th transmission bit.
Similarly, ```k: 4 latency, 379.000 clk``` means received a bit ```1``` as the 4th transmission bit.

### message_1.out
Evaluated with 14 threadblocks for Spy and Trojan. Each threadblock sending a message of ```00000000000000000000```.
The recevied transmission matches the data sent.

### message_2.out
Evaluated with 14 threadblocks for Spy and Trojan. Each threadblock sending a message of ```11111111111111111111```.
The recevied transmission matches the data sent.

### message_3.out
Evaluated with 14 threadblocks for Spy and Trojan. Each threadblock sending a message of ```0101010101```.
The recevied transmission matches the data sent.
