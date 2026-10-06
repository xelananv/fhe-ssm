# Prompt selection for the long autoregressive run (plaintext emulation; spec_decode/prompt_select_longrun.py)

ticks 172, Newton seed 0.1/12, hard filters: newtonWorst < 1e-6, msHiMax < 30.0, no repetition loop; soft: nearTies (margin < 0.05), distinct-2

| class | candidates | pass newton | pass ms/hi | pass loop | pass all | selected | nearTies of the selected (min..max) | msHiMax of the selected (max) |
|---|---|---|---|---|---|---|---|---|
| A | 160 | 160 | 160 | 157 | 157 | 16 | 0..3 | 14.39 |
| B | 160 | 160 | 160 | 155 | 155 | 16 | 0..2 | 8.63 |
| C | 160 | 160 | 160 | 153 | 153 | 16 | 0..1 | 1.35 |
| D | 160 | 160 | 160 | 149 | 149 | 16 | 0..1 | 15.86 |

selected 64 prompts -> `longrun_prompts64_v2.txt` (interleaved A,B,C,D)

| lane | class | prompt tokens | generated | nearTies | minMargin | msHiMax (site) | max abs state | distinct-2 | prompt (start) | generation (start) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | A | 65 | 108 | 0 | 0.0699 | 0.97 (L0.cm) | 2.41 | 0.5234 | 'Scientists have long sought to develop a theory that can des' | '\nThe most popular theory is the Standard Model, which describes the fundamental ' |
| 1 | B | 47 | 126 | 0 | 0.0588 | 0.89 (L1.tm) | 3.51 | 0.088 | "Excel's Accounting number format adds decimal to the value i" | '\n- Excel 2010: How to use the Excel 2010 functions\n- Excel 2010: How to use the ' |
| 2 | C | 11 | 162 | 0 | 0.0862 | 0.93 (L1.cm) | 2.77 | 0.1677 | 'A constitution is a document that outlines the founding prin' | ' govern the government of a country. The constitution is the supreme law of the ' |
| 3 | D | 7 | 166 | 0 | 0.0847 | 1.00 (L4.tm) | 2.84 | 0.2545 | 'The scientists discovered a new species of' | ' sea anemone, which they named Eurydice, after the Greek goddess of the sea.\nThe' |
| 4 | A | 70 | 103 | 0 | 0.0616 | 0.93 (L1.tm) | 2.84 | 0.2745 | 'Man and metals have an age-old relationship. Different perio' | '\nThe word ‘gold’ is derived from the Sanskrit word ‘guru’ which means ‘to be a t' |
| 5 | B | 35 | 138 | 1 | 0.0451 | 0.92 (L0.cm) | 2.90 | 0.4891 | 'The breed derives its name from the place that served as its' | ' It was founded in the early Middle Ages by the Normans, who were driven out of ' |
| 6 | C | 12 | 161 | 1 | 0.0149 | 1.02 (L0.cm) | 2.22 | 0.3625 | "Because ''Uncle Tom's Cabin'' was a work" | ' of fiction, it is\r\n\r\n      not a work of fiction. It is a true story, and it is' |
| 7 | D | 3 | 170 | 0 | 0.0784 | 0.89 (L1.tm) | 2.78 | 0.1834 | 'The sea is' | ' a great place to go for a swim, but it is not a place to go to sleep.\nThe ocean' |
| 8 | A | 72 | 101 | 1 | 0.0212 | 1.48 (L6.tm) | 2.82 | 0.49 | 'Frederick Douglass (born Frederick Augustus Washington Baile' | '\nBorn in New York City, he was the son of a slave, and was educated at the publi' |
| 9 | B | 36 | 137 | 1 | 0.0297 | 0.90 (L3.tm) | 3.56 | 0.2721 | 'Adult dragonflies typically hold a small territory, which is' | '\nThe adult dragonfly is a large, dark brown or black butterfly with a black body' |
| 10 | C | 12 | 161 | 1 | 0.0435 | 1.30 (L3.tm) | 3.77 | 0.3312 | 'Explain how The Baroque Period was different than The Classi' | ' Period.\n1. The Renaissance was a period of great change in the world.\n2. The Re' |
| 11 | D | 8 | 165 | 0 | 0.0843 | 0.88 (L0.cm) | 3.78 | 0.1707 | 'A noun is a word that names a' | ' person, place, thing, or idea.\n- A noun is a word that describes a person, plac' |
| 12 | A | 77 | 96 | 1 | 0.0445 | 14.39 (L16.tm) | 2.88 | 0.1895 | 'Warning is hereby given that not all Project Ideas are appro' | '<|endoftext|>The first step in the process of learning is to learn to read. Read' |
| 13 | B | 34 | 139 | 1 | 0.0037 | 1.03 (L1.cm) | 2.16 | 0.1087 | 'The sixth century abbot, Dionysius Exiguus, created the cale' | ' The year was divided into 12 months, and the year was divided into 12 months. T' |
| 14 | C | 15 | 158 | 1 | 0.0021 | 1.07 (L1.cm) | 2.30 | 0.3057 | 'The calendar we use today to track days, months and years is' | ' on the Gregorian calendar, which is based on the rotation of the Earth. The Gre' |
| 15 | D | 8 | 165 | 0 | 0.0846 | 0.83 (L21.tm) | 3.07 | 0.1402 | 'Photosynthesis is the process by which plants' | ' convert sunlight into chemical energy. The process is the process by which ener' |
| 16 | A | 79 | 94 | 1 | 0.0196 | 1.05 (L4.cm) | 3.53 | 0.0538 | 'Forgetting (retention loss) refers to apparent loss of infor' | '\n- Memory of events\n- Memory of events\n- Memory of events\n- Memory of events\n- M' |
| 17 | B | 33 | 140 | 1 | 0.0454 | 1.05 (ln_out) | 2.68 | 0.1007 | 'States of matter section 4 key ideas 〉what are some properti' | '\n3.1.1 Describe the properties of gases.\n2.1.1 Describe the properties of gases.' |
| 18 | C | 15 | 158 | 1 | 0.0039 | 0.90 (L1.cm) | 3.73 | 0.2611 | 'This lesson uses Jane Addams Award-winning books to explore ' | ', voice, and voice.\n- Grades: PreK–K, 1–2, 3–5, 6–8, 9–12\n- Lesson Plan Type: Le' |
| 19 | D | 9 | 164 | 0 | 0.0853 | 1.30 (L21.tm) | 2.56 | 0.135 | 'Leonardo da Vinci painted the Mona' | ' Lisa in 1452.\nThe Mona Lisa is one of the most famous paintings in the world. I' |
| 20 | A | 73 | 100 | 2 | 0.0246 | 0.95 (L0.cm) | 2.60 | 0.6162 | 'More than one million tiny weed-eating beetles have been rel' | '\nThe beetles are native to the Great Lakes region, and have been spreading north' |
| 21 | B | 46 | 127 | 1 | 0.0350 | 0.97 (L4.cm) | 2.49 | 0.0794 | 'This section is a pre-reading activity to prep students for ' | '\n- What is the purpose of the song?\n- What is the purpose of the song?\n- What is' |
| 22 | C | 13 | 160 | 1 | 0.0242 | 0.92 (L0.cm) | 3.57 | 0.2264 | 'Discuss who Nicholas de Grandmaison was and why he was' | ' so important to the French.\n- What was the name of the man who was the first to' |
| 23 | D | 4 | 169 | 0 | 0.0921 | 0.82 (L1.cm) | 3.58 | 0.125 | 'First, second,' | ' and third-person plural.\n- Third person plural.\n- Third person plural.\n- Third ' |
| 24 | A | 71 | 102 | 2 | 0.0353 | 0.96 (L1.cm) | 2.73 | 0.4158 | 'Dengue comes in two forms. Dengue fever usually starts with ' | ' Dengue fever is transmitted by mosquitoes, and is transmitted by mosquitoes.\nDe' |
| 25 | B | 33 | 140 | 1 | 0.0249 | 1.13 (L2.cm) | 3.39 | 0.0719 | 'Nutrient availability is primarily determined by soil textur' | ' Soil texture is the texture of the soil. Soil texture is the texture of the soi' |
| 26 | C | 10 | 163 | 1 | 0.0397 | 1.35 (L22.tm) | 3.48 | 0.216 | 'Sirius B is not a normal star; its' | ' surface temperature is about 5,000 degrees Fahrenheit, and it is about the size' |
| 27 | D | 7 | 166 | 0 | 0.0563 | 0.97 (L4.cm) | 2.60 | 0.1212 | 'The Nile is the longest river in' | ' the world, and the Nile River is the largest river in the world. It is the larg' |
| 28 | A | 70 | 103 | 2 | 0.0173 | 1.11 (L1.cm) | 2.87 | 0.2157 | 'The aim of this course is to provide students with a deeper ' | '\nThe course will be taught in English.\n- Course title: Linguistics\n- Course dura' |
| 29 | B | 37 | 136 | 2 | 0.0237 | 2.26 (L22.tm) | 2.36 | 0.6741 | 'To allow them to study antibiotics under more realistic cond' | '\nThe researchers found that the bacteria could be used to control the expression' |
| 30 | C | 9 | 164 | 1 | 0.0468 | 0.83 (L0.cm) | 2.59 | 0.184 | 'Water is our most essential nutrient. We can' | '’t live without it.\nWater is essential for life. It is the most abundant element' |
| 31 | D | 3 | 170 | 0 | 0.1003 | 0.99 (L1.tm) | 2.77 | 0.1124 | 'A triangle has' | ' four sides, and the sum of the angles is 180°.\nThe sum of the angles of a trian' |
| 32 | A | 68 | 105 | 2 | 0.0014 | 0.99 (L4.tm) | 3.63 | 0.2019 | 'Session two of this project focused on construction and the ' | '\nThe class also explored the use of materials to create a variety of objects, in' |
| 33 | B | 44 | 129 | 2 | 0.0094 | 8.63 (L16.tm) | 2.52 | 0.5469 | 'Several food items are thought to help in preventing brain c' | '\nThe study, published in the journal Cancer Research, found that people who ate ' |
| 34 | C | 10 | 163 | 1 | 0.0446 | 0.91 (L0.cm) | 3.17 | 0.1667 | 'However, for all of these machines, only integer' | ' numbers are used.\nThe first two numbers are the number of bits in the number, a' |
| 35 | D | 7 | 166 | 0 | 0.0534 | 0.82 (L0.cm) | 2.84 | 0.097 | 'A triangle is a shape with three' | ' sides.\n- A triangle is a figure with four sides.\n- A triangle is a figure with ' |
| 36 | A | 72 | 101 | 3 | 0.0234 | 1.52 (L18.tm) | 2.51 | 0.79 | 'Lymphedema is caused by a blockage in your lymphatic system,' | '\nLymphedema can occur in any part of the body, but it is most common in the arms' |
| 37 | B | 38 | 135 | 2 | 0.0305 | 0.99 (L0.cm) | 2.28 | 0.3507 | 'The moon is aged around 4.51 billion years after chronologic' | '\nThe age of the moon is estimated to be about 4.5 billion years, which is about ' |
| 38 | C | 9 | 164 | 1 | 0.0356 | 0.91 (L0.cm) | 3.07 | 0.1595 | 'Definition, usage and a list of comparison examples' | '.\nA list of related documents that contains the word "conclusion" in each chapte' |
| 39 | D | 11 | 162 | 0 | 0.1385 | 1.92 (L4.tm) | 4.09 | 0.0932 | 'The planets of the solar system are Mercury, Venus,' | ' Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Uranus, Neptune, Uranus, Neptune' |
| 40 | A | 66 | 107 | 3 | 0.0030 | 1.05 (L18.tm) | 2.60 | 0.6981 | 'Constitution of the united states we the people of the unite' | '\nThe American History Sourcebook is a comprehensive resource for students, educa' |
| 41 | B | 38 | 135 | 2 | 0.0014 | 0.99 (L0.cm) | 2.49 | 0.2985 | 'The rule of thirds calls for every photo to be divided into ' | '\nThe first rule is to use the same size of paper as the photo. The second rule i' |
| 42 | C | 10 | 163 | 1 | 0.0193 | 0.82 (L1.tm) | 3.48 | 0.1543 | 'From the formal paintings on tombs, the Egyptian' | ' tomb paintings are the most beautiful and the most beautiful. The paintings are' |
| 43 | D | 6 | 167 | 0 | 0.1072 | 1.92 (L4.tm) | 4.06 | 0.0904 | 'Mercury, Venus,' | ' Earth, Mars, Jupiter, Saturn, Uranus, Neptune, Uranus, Neptune, Uranus, Neptune' |
| 44 | A | 66 | 107 | 3 | 0.0099 | 1.14 (L20.tm) | 2.43 | 0.5094 | 'Arduino is an open-source prototyping platform based on easy' | '\nThe Arduino is a microcontroller-based microcontroller, which means that it can' |
| 45 | B | 39 | 134 | 2 | 0.0223 | 0.99 (L0.cm) | 2.29 | 0.2782 | 'When interest rates go down, it becomes cheaper to borrow mo' | '\nThe same thing happens when the economy is in a recession. When the economy is ' |
| 46 | C | 10 | 163 | 1 | 0.0265 | 0.83 (L0.cm) | 2.36 | 0.142 | 'The 36 Get Ready to Read! skill-building' | ' activities are designed to help children develop the skills they need to succee' |
| 47 | D | 3 | 170 | 0 | 0.0530 | 0.84 (L0.cm) | 3.04 | 0.0828 | 'The sun is' | ' the source of all life, and the sun is the source of all life.\nThe sun is the s' |
| 48 | A | 78 | 95 | 3 | 0.0378 | 1.68 (L5.tm) | 2.30 | 0.4894 | 'Originally, the Delian League headquartered on the island of' | '\nThe League of Athens was founded in the year of the Peloponnesian War, when the' |
| 49 | B | 46 | 127 | 2 | 0.0189 | 4.55 (L15.cm) | 2.08 | 0.2381 | 'Japanese plans for a seaborne invasion of Port Moresby had b' | '\nThe Japanese had been able to concentrate their forces in the area of the Solom' |
| 50 | C | 12 | 161 | 1 | 0.0033 | 0.97 (L4.tm) | 3.47 | 0.1187 | 'Let me start by graphing the two vertical asymptotes x' | '1 and x2.\nThe x-axis is the x-axis, and the y-axis is the y-axis.\nThe y-axis is ' |
| 51 | D | 3 | 170 | 0 | 0.0645 | 0.98 (L1.cm) | 3.73 | 0.0533 | 'The largest ocean' | ' basins are the Atlantic Ocean, the Indian Ocean, the Indian Ocean, the Indian O' |
| 52 | A | 78 | 95 | 3 | 0.0095 | 1.48 (L4.cm) | 2.25 | 0.4468 | 'Chemistry is the science of matter, the physical material th' | '\nThe atoms of matter are made up of atoms, which are made up of protons, neutron' |
| 53 | B | 33 | 140 | 2 | 0.0074 | 0.99 (L1.tm) | 2.55 | 0.2374 | 'The bulk modulus describes how a substance reacts when squee' | ' If the surface is stretched, the surface tension is equal to the surface tensio' |
| 54 | C | 12 | 161 | 1 | 0.0475 | 0.91 (L0.cm) | 2.66 | 0.1125 | 'A half-step on a piano is the distance between one' | ' note and the next.\nThe distance between the two notes is called the octave.\nThe' |
| 55 | D | 8 | 165 | 1 | 0.0467 | 0.96 (L0.tm) | 2.54 | 0.4573 | 'The old man sat by the fire and' | ' watched the\r\n\r\n      fire.\r\n\r\n    \r\n\r\n\r\n      “I have been thinking,” he said, ' |
| 56 | A | 75 | 98 | 3 | 0.0074 | 1.00 (L1.tm) | 2.97 | 0.4227 | 'The Phoebe ring was first discovered in 2009 by astrophysici' | '\nThe rings are made of a mixture of water and methane, and they are about the si' |
| 57 | B | 46 | 127 | 2 | 0.0057 | 0.90 (L18.tm) | 2.83 | 0.2143 | 'The major purpose and function of the Federal Reserve System' | ' The Board of Governors is composed of the Chairman, Vice Chairman, and Vice Cha' |
| 58 | C | 15 | 158 | 1 | 0.0454 | 1.06 (L0.cm) | 2.92 | 0.1083 | 'Racism…the belief that a particular race is superior to othe' | ' the belief that one’s own race is superior to others; the belief that one’s own' |
| 59 | D | 7 | 166 | 1 | 0.0082 | 1.06 (L1.tm) | 3.07 | 0.4 | 'The capital city of France is known' | ' as the capital of France. It is located in the south-west of the continent, in ' |
| 60 | A | 76 | 97 | 3 | 0.0265 | 0.87 (L0.cm) | 2.81 | 0.3542 | 'Community Commons has recently added data from the USDA Food' | '\nThe data in this report are based on the USDA’s National Agricultural Statistic' |
| 61 | B | 39 | 134 | 2 | 0.0011 | 1.25 (L4.cm) | 2.68 | 0.1955 | 'Acetylcholinesterase helps regulate nervous system activity ' | '\nThe enzyme is found in the saliva glands, where it is produced in the salivary ' |
| 62 | C | 12 | 161 | 1 | 0.0085 | 0.85 (ln_out) | 2.85 | 0.0938 | 'Focus on 3-5 of the most important criteria for the' | ' development of a successful project.\n- Identify the key elements of the project' |
| 63 | D | 7 | 166 | 1 | 0.0485 | 15.86 (ln_out) | 2.52 | 0.3939 | 'The castle stood on a hill overlooking' | ' the sea, and was surrounded by a\r\n\r\nfence of trees. The castle was surrounded b' |
