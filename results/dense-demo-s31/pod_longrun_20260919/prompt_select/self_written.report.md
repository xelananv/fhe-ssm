# Prompt selection for the long autoregressive run (plaintext emulation; spec_decode/prompt_select_longrun.py)

ticks 172, Newton seed 0.1/12, hard filters: newtonWorst < 1e-6, msHiMax < 30.0, no repetition loop; soft: nearTies (margin < 0.05), distinct-2

| class | candidates | pass newton | pass ms/hi | pass loop | pass all | selected | nearTies of the selected (min..max) | msHiMax of the selected (max) |
|---|---|---|---|---|---|---|---|---|
| A | 31 | 31 | 31 | 31 | 31 | 8 | 0..4 | 1.49 |
| B | 31 | 31 | 30 | 30 | 29 | 8 | 1..4 | 1.22 |
| C | 32 | 32 | 32 | 32 | 32 | 8 | 0..2 | 20.03 |
| D | 0 | 0 | 0 | 0 | 0 | 0 | -..- | 0.00 |

selected 24 prompts -> `self_written.txt` (interleaved A,B,C,D)

| lane | class | prompt tokens | generated | nearTies | minMargin | msHiMax (site) | max abs state | distinct-2 | prompt (start) | generation (start) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | A | 65 | 108 | 0 | 0.0670 | 0.91 (L0.cm) | 2.47 | 0.1121 | 'When snow falls on a city, the usual noise disappears. Cars ' | '\nThe children are playing in the snow. They are playing with the snow. They are ' |
| 1 | B | 44 | 129 | 1 | 0.0370 | 1.05 (L0.cm) | 2.41 | 0.2344 | 'Wind is simply air that is moving from one place to another.' | ' The wind is the result of this movement.\nThe wind is a force that moves air. It' |
| 2 | C | 11 | 162 | 0 | 0.0697 | 1.02 (L0.cm) | 2.88 | 0.1491 | 'One of the simplest ways to save water at home is' | ' to use a water-saving shower head.\nThe water-saving shower head is a simple, in' |
| 3 | A | 66 | 107 | 1 | 0.0394 | 0.98 (L1.cm) | 2.62 | 0.2358 | 'Rain begins with water so small that it cannot be seen. Warm' | '\nThe water vapor condenses into tiny droplets, which fall to the ground as rain.' |
| 4 | B | 34 | 139 | 2 | 0.0060 | 1.07 (L1.cm) | 2.78 | 0.2464 | 'The market closes when the sun goes down. Sellers pack their' | '\nThe birds are then released into the forest. The birds are attracted to the tre' |
| 5 | C | 12 | 161 | 1 | 0.0114 | 1.27 (L0.cm) | 2.42 | 0.4062 | 'At the edge of the forest stood a small wooden house where' | '\r\n\r\n      the old man had lived.\r\n\r\n    \r\n\r\n\r\n      “I have come to ask you to c' |
| 6 | A | 76 | 97 | 2 | 0.0172 | 1.49 (L1.tm) | 3.18 | 0.1875 | 'The mountain hut was built of grey stone with a roof of flat' | '\nThe hut was built of logs, and the roof was made of bamboo. The roof was made o' |
| 7 | B | 38 | 135 | 2 | 0.0003 | 0.93 (L0.cm) | 2.90 | 0.1791 | 'An island is a piece of land with water on every side. Some ' | '\nThe word island comes from the Latin word islandus, meaning "island." The word ' |
| 8 | C | 10 | 163 | 1 | 0.0225 | 1.73 (L9.tm) | 3.21 | 0.2593 | 'Wool keeps us warm in winter because its fibers' | ' are very fine.\nThe wool is made from the wool of the wool sheep, which is a sma' |
| 9 | A | 68 | 105 | 3 | 0.0162 | 0.99 (L0.cm) | 2.70 | 0.4327 | 'The moon does not make its own light. It shines because it r' | '\nThe moon is not the only object that changes in the same way. The moon is also ' |
| 10 | B | 39 | 134 | 2 | 0.0015 | 0.88 (L0.cm) | 3.69 | 0.1579 | 'A seed can wait for years until the conditions are right. It' | ' The seedling is a seedling, and it is a seedling.\nThe seedling is a seedling th' |
| 11 | C | 8 | 165 | 1 | 0.0318 | 1.11 (L21.cm) | 3.47 | 0.2439 | 'Before electric light, families spent their evenings' | ' in the kitchen, where they ate, slept, and slept.\nThe family was divided into t' |
| 12 | A | 71 | 102 | 3 | 0.0224 | 0.97 (L0.cm) | 3.16 | 0.3564 | 'Bread rises because of a tiny living thing called yeast. Whe' | '\nThe yeast in the dough is called the yeast. The yeast is made up of many differ' |
| 13 | B | 42 | 131 | 3 | 0.0080 | 1.22 (L0.cm) | 3.18 | 0.3615 | 'A year is the time the earth needs to travel once around the' | '\nThe year is divided into 12 months, which are called months. Each month has a s' |
| 14 | C | 11 | 162 | 1 | 0.0348 | 0.96 (L0.cm) | 2.48 | 0.205 | 'The bakery was famous across the whole region for its' | ' fine quality of bread, and the quality of its products.\nThe first bakeries in t' |
| 15 | A | 64 | 109 | 3 | 0.0023 | 0.85 (ln_out) | 2.26 | 0.2778 | 'The old library had a smell of dust, leather, and floor poli' | '\nThe library was a small, bare, wooden building, with a wooden roof. The walls w' |
| 16 | B | 37 | 136 | 3 | 0.0252 | 0.98 (L0.cm) | 2.67 | 0.1926 | 'Ice floats because it is lighter than liquid water. This sim' | '\nThe ice is a good insulator. It keeps the ice from melting, and it keeps the ic' |
| 17 | C | 11 | 162 | 2 | 0.0049 | 1.01 (L4.cm) | 2.71 | 0.441 | 'A simple experiment with a glass of water can show that' | ' the water is not only a liquid, but also a gas.\nThe experiment is simple, but i' |
| 18 | A | 65 | 108 | 4 | 0.0001 | 1.16 (L0.cm) | 2.57 | 0.7196 | 'The map was drawn by hand on a sheet of yellow paper. Rivers' | '\nThe map is in the public domain.\nThis map is in the public domain because it wa' |
| 19 | B | 41 | 132 | 3 | 0.0276 | 0.93 (L0.cm) | 2.96 | 0.1527 | 'Plants make their own food from sunlight, air, and water. Th' | '\nThe process of photosynthesis is the process by which energy is converted into ' |
| 20 | C | 9 | 164 | 2 | 0.0048 | 20.03 (ln_out) | 3.41 | 0.3374 | 'Behind the school there was a field where' | ' the\r\n\r\nfarmers were ploughing, and the children were playing in the\r\n\r\nfields.\r' |
| 21 | A | 75 | 98 | 4 | 0.0010 | 1.08 (L1.cm) | 2.72 | 0.1959 | 'Water in the valley is always moving, even when the lake loo' | '\nThe water in the lake is very salty, and the water is very salty. The water is ' |
| 22 | B | 43 | 130 | 4 | 0.0177 | 1.01 (L4.tm) | 2.57 | 0.3643 | 'Honey never spoils if it is kept in a closed jar. It contain' | '\nThe ancient Egyptians used honey to treat a wide variety of ailments, including' |
| 23 | C | 11 | 162 | 2 | 0.0126 | 0.86 (L0.cm) | 3.11 | 0.2609 | 'The reason the sky looks blue during the day is that' | ' the sun is high in the sky, so the blue light is reflected off the surface of t' |
