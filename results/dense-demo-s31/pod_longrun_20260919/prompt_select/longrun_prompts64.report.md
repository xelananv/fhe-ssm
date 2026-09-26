# Prompt selection for the long autoregressive run (plaintext emulation; spec_decode/prompt_select_longrun.py)

ticks 172, Newton seed 0.1/12, hard filters: newtonWorst < 1e-6, msHiMax < 30.0, no repetition loop; soft: nearTies (margin < 0.05), distinct-2

| class | candidates | pass newton | pass ms/hi | pass loop | pass all | selected | nearTies of the selected (min..max) | msHiMax of the selected (max) |
|---|---|---|---|---|---|---|---|---|
| A | 160 | 160 | 160 | 159 | 159 | 16 | 1..3 | 3.38 |
| B | 160 | 160 | 160 | 156 | 156 | 16 | 1..2 | 14.39 |
| C | 160 | 160 | 160 | 152 | 152 | 16 | 0..1 | 20.24 |
| D | 160 | 160 | 160 | 149 | 149 | 16 | 0..1 | 15.86 |

selected 64 prompts -> `longrun_prompts64.txt` (interleaved A,B,C,D)

| lane | class | prompt tokens | generated | nearTies | minMargin | msHiMax (site) | max abs state | distinct-2 | prompt (start) | generation (start) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | A | 123 | 50 | 1 | 0.0000 | 1.06 (L4.cm) | 2.39 | 0.4082 | 'We have learnt how electrons involve in stabilisation of an ' | ' Isotopes are atoms of the same element with different numbers of protons, neutr' |
| 1 | B | 54 | 119 | 1 | 0.0146 | 1.36 (L4.cm) | 2.80 | 0.4068 | "They bonded with other student activists. Berkeley's activis" | '\nThe first women to be elected to the US Congress were women. The first woman to' |
| 2 | C | 10 | 163 | 0 | 0.0583 | 1.26 (L1.tm) | 2.63 | 0.2716 | 'In the context of language learning, American pronunciation' | ' is often considered to be a complex process, and the study of pronunciation is ' |
| 3 | D | 7 | 166 | 0 | 0.0847 | 1.00 (L4.tm) | 2.84 | 0.2545 | 'The scientists discovered a new species of' | ' sea anemone, which they named Eurydice, after the Greek goddess of the sea.\nThe' |
| 4 | A | 130 | 43 | 1 | 0.0321 | 1.53 (L4.cm) | 3.07 | 0.2381 | 'If one is looking for its actual definition then it is as be' | '\n- Class is a class, which is a class.\n- Class is a class, which is a class.\n- C' |
| 5 | B | 41 | 132 | 1 | 0.0445 | 14.39 (L16.tm) | 2.88 | 0.4046 | 'Warning is hereby given that not all Project Ideas are appro' | ' Reading and following the safety precautions of all materials used in a project' |
| 6 | C | 14 | 159 | 0 | 0.0713 | 20.24 (ln_out) | 2.73 | 0.2595 | 'Nature—with its uncontainable power, unpredictability, and' | '\r\n\r\nuncontrollable power—is the most potent of all the forces of nature.\r\n\r\nIt i' |
| 7 | D | 3 | 170 | 0 | 0.0784 | 0.89 (L1.tm) | 2.78 | 0.1834 | 'The sea is' | ' a great place to go for a swim, but it is not a place to go to sleep.\nThe ocean' |
| 8 | A | 115 | 58 | 2 | 0.0035 | 1.00 (L1.tm) | 2.85 | 0.8947 | 'The Phoebe ring was first discovered in 2009 by astrophysici' | '\nThe ringed planet is a little bigger than the Earth, but it’s still bigger than' |
| 9 | B | 59 | 114 | 1 | 0.0173 | 0.86 (L0.cm) | 3.09 | 0.3274 | "It's important to encourage student involvement in both arts" | '\nThe robot arm is a simple, inexpensive, and inexpensive tool that can be used t' |
| 10 | C | 11 | 162 | 0 | 0.0862 | 0.93 (L1.cm) | 2.77 | 0.1677 | 'A constitution is a document that outlines the founding prin' | ' govern the government of a country. The constitution is the supreme law of the ' |
| 11 | D | 8 | 165 | 0 | 0.0843 | 0.88 (L0.cm) | 3.78 | 0.1707 | 'A noun is a word that names a' | ' person, place, thing, or idea.\n- A noun is a word that describes a person, plac' |
| 12 | A | 126 | 47 | 2 | 0.0112 | 1.39 (ln_out) | 2.97 | 0.7174 | 'Exemplar ap us history the era of good feelings and jackson ' | '\nThe jacksonian period is a period of history in which the United States was fou' |
| 13 | B | 60 | 113 | 1 | 0.0413 | 1.13 (L3.tm) | 2.75 | 0.2054 | 'Linnaeus attempted to name organisms in a way that would mak' | '\nThe Linnaean system was based on the idea that all living things were related t' |
| 14 | C | 13 | 160 | 0 | 0.0547 | 0.91 (L4.cm) | 3.62 | 0.1509 | 'Several food items are thought to help in preventing brain c' | ' the health of the brain.\nThe following foods are considered to be brain-healthy' |
| 15 | D | 8 | 165 | 0 | 0.0846 | 0.83 (L21.tm) | 3.07 | 0.1402 | 'Photosynthesis is the process by which plants' | ' convert sunlight into chemical energy. The process is the process by which ener' |
| 16 | A | 122 | 51 | 2 | 0.0011 | 3.38 (L9.tm) | 2.22 | 0.68 | 'A young Scotsman engaged in the fur trade out of Montreal, M' | '\nIn 1805, Mackenzie returned to Canada, where he was appointed to the post of su' |
| 17 | B | 50 | 123 | 1 | 0.0031 | 0.89 (L1.tm) | 3.27 | 0.0984 | 'Discuss who Nicholas de Grandmaison was and why he was inter' | '\n- What is the relationship between the two men?\n- What is the relationship betw' |
| 18 | C | 11 | 162 | 0 | 0.0565 | 1.04 (L0.cm) | 3.18 | 0.1304 | 'Sight, sound, and touch are the senses we' | ' use to communicate with others.\n- The senses are the senses that we use to unde' |
| 19 | D | 9 | 164 | 0 | 0.0853 | 1.30 (L21.tm) | 2.56 | 0.135 | 'Leonardo da Vinci painted the Mona' | ' Lisa in 1452.\nThe Mona Lisa is one of the most famous paintings in the world. I' |
| 20 | A | 118 | 55 | 2 | 0.0047 | 1.17 (L6.tm) | 2.33 | 0.6481 | 'The breed derives its name from the place that served as its' | '\nThe first inhabitants of the area were the Gauls, who were probably of Gaulish ' |
| 21 | B | 40 | 133 | 2 | 0.0133 | 0.99 (L0.cm) | 1.99 | 0.5682 | 'Mary Berbee, a professor in the department of botany, is res' | '\n“We’re trying to get people to think about the way they use their environment t' |
| 22 | C | 9 | 164 | 0 | 0.0507 | 0.82 (L4.tm) | 3.10 | 0.1227 | 'Third, enforce the idea that sometimes writing can' | ' be a tool for social change.\n- Make sure that the writing is relevant to the au' |
| 23 | D | 4 | 169 | 0 | 0.0921 | 0.82 (L1.cm) | 3.58 | 0.125 | 'First, second,' | ' and third-person plural.\n- Third person plural.\n- Third person plural.\n- Third ' |
| 24 | A | 120 | 53 | 2 | 0.0002 | 1.48 (L6.tm) | 2.25 | 0.5769 | 'Frederick Douglass (born Frederick Augustus Washington Baile' | '\nBaxter was born in Philadelphia, Pennsylvania, and was educated at the Universi' |
| 25 | B | 58 | 115 | 2 | 0.0430 | 0.90 (L1.tm) | 2.29 | 0.5 | 'When interest rates go down, it becomes cheaper to borrow mo' | '\nBut the problem is that the economy is not growing as fast as the demand for mo' |
| 26 | C | 9 | 164 | 0 | 0.0609 | 1.50 (L22.tm) | 2.37 | 0.1227 | 'According to the U.S. Department of' | ' Energy, the average American consumes about 2,000 calories a day, and the avera' |
| 27 | D | 7 | 166 | 0 | 0.0563 | 0.97 (L4.cm) | 2.60 | 0.1212 | 'The Nile is the longest river in' | ' the world, and the Nile River is the largest river in the world. It is the larg' |
| 28 | A | 127 | 46 | 2 | 0.0354 | 0.89 (L1.cm) | 1.84 | 0.5556 | 'Freedman shows Lincoln as a human being. Lincoln was sometim' | '\nThe president was a man of great intelligence, but he was also a man of great h' |
| 29 | B | 53 | 120 | 2 | 0.0079 | 0.97 (L0.cm) | 2.58 | 0.3613 | 'Scientists have long sought to develop a theory that can des' | '\nThe theory of general relativity is the most widely accepted theory of the univ' |
| 30 | C | 11 | 162 | 0 | 0.0801 | 0.88 (L0.cm) | 2.55 | 0.1056 | 'The 5th Grade learned about different Native American tribes' | ' lived in the area. They learned about the different tribes that lived in the ar' |
| 31 | D | 3 | 170 | 0 | 0.1003 | 0.99 (L1.tm) | 2.77 | 0.1124 | 'A triangle has' | ' four sides, and the sum of the angles is 180°.\nThe sum of the angles of a trian' |
| 32 | A | 101 | 72 | 2 | 0.0070 | 1.54 (L1.cm) | 2.19 | 0.5352 | 'So soon as Galileo had completed his elementary education, h' | '\nIn 1620, Galileo was appointed to the University of Padua, where he studied mat' |
| 33 | B | 52 | 121 | 2 | 0.0035 | 0.97 (L1.cm) | 2.24 | 0.3167 | 'People have used petroleum for thousands of years, for a var' | '\nThe first known use of petroleum was in the early 19th century, when it was use' |
| 34 | C | 13 | 160 | 1 | 0.0149 | 1.02 (L0.cm) | 2.22 | 0.3648 | "Because ''Uncle Tom's Cabin'' was a work of" | ' fiction, it is\r\n\r\n      not a work of fiction. It is a true story, and it is tr' |
| 35 | D | 7 | 166 | 0 | 0.0534 | 0.82 (L0.cm) | 2.84 | 0.097 | 'A triangle is a shape with three' | ' sides.\n- A triangle is a figure with four sides.\n- A triangle is a figure with ' |
| 36 | A | 129 | 44 | 2 | 0.0366 | 1.71 (ln_out) | 2.23 | 0.5349 | 'The drag coefficient is a common measure in automotive desig' | ' One of the most common ways to reduce drag is to reduce the mass of the vehicle' |
| 37 | B | 49 | 124 | 2 | 0.0006 | 1.52 (ln_out) | 2.37 | 0.3089 | 'Non-numeric characters dollar signs, percent signs, commas, ' | '\n- The first thing to do is to find the number of the word in the word bank.\n- T' |
| 38 | C | 12 | 161 | 1 | 0.0327 | 0.85 (L0.cm) | 2.51 | 0.3438 | 'Students from Grades 1-8 learn Mathematics using JUMP' | ', a program that allows them to create and manipulate graphs, charts, and graphs' |
| 39 | D | 11 | 162 | 0 | 0.1385 | 1.92 (L4.tm) | 4.09 | 0.0932 | 'The planets of the solar system are Mercury, Venus,' | ' Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Uranus, Neptune, Uranus, Neptune' |
| 40 | A | 121 | 52 | 2 | 0.0135 | 1.05 (L4.cm) | 2.46 | 0.4314 | 'Forgetting (retention loss) refers to apparent loss of infor' | ' The amount of time spent in the process of memory storage is called the "memory' |
| 41 | B | 51 | 122 | 2 | 0.0147 | 1.28 (L1.tm) | 2.98 | 0.2645 | 'Over the centuries, paper has been made from a wide variety ' | '\nThe first paper was made from the bark of the cedar tree, which was harvested i' |
| 42 | C | 12 | 161 | 1 | 0.0435 | 1.30 (L3.tm) | 3.77 | 0.3312 | 'Explain how The Baroque Period was different than The Classi' | ' Period.\n1. The Renaissance was a period of great change in the world.\n2. The Re' |
| 43 | D | 6 | 167 | 0 | 0.1072 | 1.92 (L4.tm) | 4.06 | 0.0904 | 'Mercury, Venus,' | ' Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Uranus, Neptune, Uranus, Neptune' |
| 44 | A | 105 | 68 | 2 | 0.0160 | 2.15 (ln_out) | 2.63 | 0.2836 | "Sensitivity is the measure of the loudspeaker's ability to c" | '\nThe sound power level is measured in decibels (dB). The sound power level is ex' |
| 45 | B | 47 | 126 | 2 | 0.0178 | 1.19 (ln_out) | 2.08 | 0.256 | 'In all, we have studied vocabulary learning with more than 2' | '\nWe have found that the most effective instructional strategies are those that a' |
| 46 | C | 13 | 160 | 1 | 0.0423 | 1.00 (L0.cm) | 2.51 | 0.3208 | 'The rule of thirds calls for every photo to be divided into' | ' two parts, one for each photo, and the other half to be divided into two parts.' |
| 47 | D | 3 | 170 | 0 | 0.0530 | 0.84 (L0.cm) | 3.04 | 0.0828 | 'The sun is' | ' the source of all life, and the sun is the source of all life.\nThe sun is the s' |
| 48 | A | 117 | 56 | 3 | 0.0090 | 1.07 (L0.cm) | 2.38 | 0.8909 | 'Atrial septal defect (ASD) is a hole in the wall between the' | '\nAortic valve disease is a condition in which the valve between the heart and th' |
| 49 | B | 46 | 127 | 2 | 0.0189 | 4.55 (L15.cm) | 2.08 | 0.2381 | 'Japanese plans for a seaborne invasion of Port Moresby had b' | '\nThe Japanese had been able to concentrate their forces in the area of the Solom' |
| 50 | C | 13 | 160 | 1 | 0.0432 | 0.88 (L1.tm) | 3.53 | 0.2579 | 'All materials fall into one of three classifications when it' | ' from one state to another:\n- Transient: The material is in a state of transitio' |
| 51 | D | 3 | 170 | 0 | 0.0645 | 0.98 (L1.cm) | 3.73 | 0.0533 | 'The largest ocean' | ' basins are the Atlantic Ocean, the Indian Ocean, the Indian Ocean, the Indian O' |
| 52 | A | 129 | 44 | 3 | 0.0172 | 1.26 (L1.tm) | 1.97 | 0.7907 | 'The story of Jonah has been viewed and interpreted in three ' | '\nThe first part of the story is that Jonah was a prophet, and that he was sent t' |
| 53 | B | 52 | 121 | 2 | 0.0203 | 0.99 (L0.cm) | 2.44 | 0.225 | 'In this center, students pick 5 word cards and use those 5 w' | '\nThis is a great way to introduce the concept of the sentence.\nThis is a great w' |
| 54 | C | 9 | 164 | 1 | 0.0468 | 0.83 (L0.cm) | 2.59 | 0.184 | 'Water is our most essential nutrient. We can' | '’t live without it.\nWater is essential for life. It is the most abundant element' |
| 55 | D | 8 | 165 | 1 | 0.0467 | 0.96 (L0.tm) | 2.54 | 0.4573 | 'The old man sat by the fire and' | ' watched the\r\n\r\n      fire.\r\n\r\n    \r\n\r\n\r\n      “I have been thinking,” he said, ' |
| 56 | A | 122 | 51 | 3 | 0.0057 | 1.16 (L1.tm) | 2.31 | 0.66 | 'On this date in 1751, James Madison was born in Virginia. Th' | '\nIn 1787, Madison was elected to the Continental Congress, where he served as a ' |
| 57 | B | 52 | 121 | 2 | 0.0148 | 1.09 (L1.cm) | 2.62 | 0.2167 | 'Cirrocumulus is the formation of clouds in the sky and is re' | '\nThe formation of cirrus clouds is caused by the formation of water droplets in ' |
| 58 | C | 13 | 160 | 1 | 0.0175 | 0.90 (L0.cm) | 2.95 | 0.1824 | 'They all have a stationary phase (a solid, or a liquid' | ', or a gas), and they all have a phase transition.\nThe phase transition is the p' |
| 59 | D | 7 | 166 | 1 | 0.0082 | 1.06 (L1.tm) | 3.07 | 0.4 | 'The capital city of France is known' | ' as the capital of France. It is located in the south-west of the continent, in ' |
| 60 | A | 128 | 45 | 3 | 0.0009 | 1.60 (L4.cm) | 2.12 | 0.6364 | 'This lesson uses Jane Addams Award-winning books to explore ' | '\nThis resource is part of a larger resource on the author, Jane Addams, The Life' |
| 61 | B | 46 | 127 | 2 | 0.0057 | 0.90 (L18.tm) | 2.83 | 0.2143 | 'The major purpose and function of the Federal Reserve System' | ' The Board of Governors is composed of the Chairman, Vice Chairman, and Vice Cha' |
| 62 | C | 10 | 163 | 1 | 0.0446 | 0.91 (L0.cm) | 3.17 | 0.1667 | 'However, for all of these machines, only integer' | ' numbers are used.\nThe first two numbers are the number of bits in the number, a' |
| 63 | D | 7 | 166 | 1 | 0.0485 | 15.86 (ln_out) | 2.52 | 0.3939 | 'The castle stood on a hill overlooking' | ' the sea, and was surrounded by a\r\n\r\nfence of trees. The castle was surrounded b' |
