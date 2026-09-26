# Prompt selection for the long autoregressive run (plaintext emulation; spec_decode/prompt_select_longrun.py)

ticks 172, Newton seed 0.1/12, hard filters: newtonWorst < 1e-6, msHiMax < 30.0, no repetition loop; soft: nearTies (margin < 0.05), distinct-2

| class | candidates | pass newton | pass ms/hi | pass loop | pass all | selected | nearTies of the selected (min..max) | msHiMax of the selected (max) |
|---|---|---|---|---|---|---|---|---|
| A | 31 | 31 | 31 | 31 | 31 | 8 | 5..8 | 1.66 |
| B | 31 | 31 | 31 | 31 | 31 | 8 | 3..5 | 24.65 |
| C | 32 | 32 | 32 | 32 | 32 | 8 | 4..8 | 8.55 |
| D | 0 | 0 | 0 | 0 | 0 | 0 | -..- | 0.00 |

selected 24 prompts -> `pol_self.txt` (interleaved A,B,C,D)

| lane | class | prompt tokens | generated | nearTies | minMargin | msHiMax (site) | max abs state | distinct-2 | prompt (start) | generation (start) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | A | 71 | 102 | 5 | 0.0162 | 1.00 (L0.cm) | 2.85 | 0.9703 | 'Bread rises because of a tiny living thing called yeast. Whe' | '\nThe process of baking is called fermentation. It takes place when yeast (or bac' |
| 1 | B | 37 | 136 | 3 | 0.0213 | 1.24 (L0.cm) | 2.38 | 0.8667 | 'A compass needle always points toward the north. It works be' | '\nThe magnetic field of Earth is caused by the movement of the core, which is mad' |
| 2 | C | 12 | 161 | 4 | 0.0002 | 4.28 (ln_out) | 2.51 | 0.825 | 'At the edge of the forest stood a small wooden house where' | '\r\n\r\n      the old man had lived.\r\n\r\n    \r\n\r\n\r\n      “I have come to ask you,” sa' |
| 3 | A | 76 | 97 | 6 | 0.0027 | 1.66 (ln_out) | 2.29 | 0.9792 | 'A honeybee may visit hundreds of flowers in a single day. Sh' | '\nThe bee’s life cycle begins when she lays her eggs on a flower or plant and the' |
| 4 | B | 39 | 134 | 4 | 0.0034 | 1.06 (L0.cm) | 2.63 | 0.9398 | 'A seed can wait for years until the conditions are right. It' | ' The plant is ready to germinate.\nThe first step in growing a tree from seeds is' |
| 5 | C | 11 | 162 | 4 | 0.0065 | 8.55 (ln_out) | 3.05 | 0.7826 | 'When she opened the door of the workshop, she found' | '\r\n\r\n      that the man was a young man.\r\n\r\n    \r\n\r\n\r\n      “What is it?” he aske' |
| 6 | A | 68 | 105 | 6 | 0.0179 | 1.05 (L1.tm) | 2.31 | 0.9519 | 'At dawn the harbor is the busiest place in the town. Boats t' | '\nThe next morning, the fishermen are busy preparing a meal for the guests. They ' |
| 7 | B | 37 | 136 | 4 | 0.0086 | 1.08 (L1.tm) | 2.50 | 0.9259 | 'Ice floats because it is lighter than liquid water. This sim' | '\nThe most important thing about this phenomenon is that it happens at night, whe' |
| 8 | C | 13 | 160 | 5 | 0.0006 | 5.19 (ln_out) | 1.69 | 0.8931 | 'When the explorers reached the top of the pass, they saw' | ' a great number of Indians\r\n\r\n      standing on the summit. They were all armed ' |
| 9 | A | 74 | 99 | 6 | 0.0110 | 1.39 (L5.tm) | 2.33 | 0.949 | 'A rainforest is built in layers, like the floors of a tall h' | '\nThe rainforest is home to many different kinds of trees, but there are only two' |
| 10 | B | 34 | 139 | 5 | 0.0075 | 0.97 (L0.cm) | 2.35 | 0.942 | 'The painter mixed her colors on a piece of broken glass. She' | '\nIn the early morning, the woman stood in front of the painting and began to pai' |
| 11 | C | 10 | 163 | 6 | 0.0085 | 1.73 (L9.tm) | 3.03 | 0.9074 | 'Wool keeps us warm in winter because its fibers' | ' are very fine.\nThe wool is made from the down of sheep, and it has a soft textu' |
| 12 | A | 65 | 108 | 6 | 0.0143 | 1.00 (L0.cm) | 2.19 | 0.9439 | 'The night train left the city a few minutes after ten. Most ' | '\nThe next morning the train stopped at a station, where it was met by a crowd of' |
| 13 | B | 41 | 132 | 5 | 0.0027 | 1.54 (L2.cm) | 2.66 | 0.9237 | 'Every language has words that are hard to translate. Some de' | '\nThe word “sad” comes from the Latin sardis (a word meaning “to be sad”). The wo' |
| 14 | C | 9 | 164 | 6 | 0.0061 | 1.01 (L1.cm) | 2.05 | 0.9018 | 'Every morning before school, the two brothers would' | ' go to the library and read.\n“I’d sit down with them and talk about what they we' |
| 15 | A | 75 | 98 | 7 | 0.0027 | 1.11 (L0.cm) | 2.55 | 0.9381 | 'Water in the valley is always moving, even when the lake loo' | '\nThe mountains are very important in the life cycle of a mountain. They form fro' |
| 16 | B | 41 | 132 | 5 | 0.0029 | 1.02 (L1.cm) | 2.38 | 0.916 | 'Bones are lighter than they look, but they are very strong. ' | '\nThe bones in your skeleton help you move and support your whole body. They also' |
| 17 | C | 11 | 162 | 7 | 0.0005 | 1.21 (L4.cm) | 2.31 | 0.9006 | 'In ancient times, travelers found their way at night by' | ' the use of a compass. The first known map showing the position of the sun was m' |
| 18 | A | 64 | 109 | 7 | 0.0039 | 0.96 (L1.tm) | 2.10 | 0.9352 | 'The old library had a smell of dust, leather, and floor poli' | '\nThe house was empty. The walls were covered with dust, but there was no sign of' |
| 19 | B | 36 | 137 | 5 | 0.0148 | 1.14 (L22.tm) | 2.54 | 0.8897 | 'Salt has been valuable for most of human history. It keeps f' | '\nThe word “salt” comes from the Latin word salus (water). The word is related to' |
| 20 | C | 9 | 164 | 7 | 0.0163 | 1.09 (L5.tm) | 2.71 | 0.8957 | 'The main difference between weather and climate is that' | ' the former are more stable, while the latter change over time.\nTemperature: The' |
| 21 | A | 67 | 106 | 8 | 0.0005 | 1.34 (L22.tm) | 2.81 | 0.981 | 'Where a great river meets the sea, it slows down and drops t' | '\nThe river that flows from the sea to the land is called the Nile River. It runs' |
| 22 | B | 35 | 138 | 5 | 0.0035 | 24.65 (ln_out) | 1.94 | 0.854 | 'The letter arrived three weeks after it was sent. The paper ' | '\n“I am very sorry, miss,” she said to me, “but I have no time to write.”\r\nShe lo' |
| 23 | C | 10 | 163 | 8 | 0.0052 | 0.97 (L0.cm) | 2.08 | 0.9691 | 'The most important thing to know about rivers is that' | ' they are not static, but constantly changing. They change in size and shape as ' |
