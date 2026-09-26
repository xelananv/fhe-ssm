# Prompt selection for the long autoregressive run (plaintext emulation; spec_decode/prompt_select_longrun.py)

ticks 172, Newton seed 0.1/12, hard filters: newtonWorst < 1e-6, msHiMax < 30.0, no repetition loop; soft: nearTies (margin < 0.05), distinct-2

| class | candidates | pass newton | pass ms/hi | pass loop | pass all | selected | nearTies of the selected (min..max) | msHiMax of the selected (max) |
|---|---|---|---|---|---|---|---|---|
| A | 160 | 160 | 160 | 160 | 160 | 16 | 1..5 | 2.03 |
| B | 160 | 160 | 160 | 160 | 160 | 16 | 2..5 | 1.87 |
| C | 160 | 160 | 160 | 160 | 160 | 16 | 2..5 | 4.08 |
| D | 160 | 160 | 160 | 160 | 160 | 16 | 1..4 | 22.89 |

selected 64 prompts -> `pol_corpus.txt` (interleaved A,B,C,D)

| lane | class | prompt tokens | generated | nearTies | minMargin | msHiMax (site) | max abs state | distinct-2 | prompt (start) | generation (start) |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | A | 77 | 96 | 1 | 0.0085 | 1.90 (L4.cm) | 1.96 | 0.9895 | 'The different responses of the two species probably reflect ' | '\n“We think it is possible that these animals are using their sense of hearing to' |
| 1 | B | 45 | 128 | 2 | 0.0031 | 1.65 (L4.cm) | 2.48 | 0.9528 | 'The most common airway problems in children are upper and lo' | '\nChildren with these conditions may have a cough that is worse at night or when ' |
| 2 | C | 10 | 163 | 2 | 0.0127 | 0.98 (L21.tm) | 2.45 | 0.9012 | 'Repeated reading can be used in a variety of' | ' ways, including as an aid to memory.\n- The ability to recall information is enh' |
| 3 | D | 3 | 170 | 1 | 0.0306 | 1.04 (L1.tm) | 2.64 | 0.9172 | 'The largest ocean' | ' basins are the Atlantic Ocean, the Indian Ocean and the Pacific Ocean. The ocea' |
| 4 | A | 70 | 103 | 2 | 0.0088 | 1.15 (L1.tm) | 1.98 | 1.0 | 'Every parent can use a little help now and then, and birds a' | '\nNow, a team of scientists at the University of California, Santa Barbara has fo' |
| 5 | B | 48 | 125 | 2 | 0.0082 | 0.98 (L0.cm) | 2.32 | 0.871 | 'Objectives for Learning: 1. Describe how Mendeleev predicted' | 'M.S. periodic table are related to each other 3. Identify the different chemical' |
| 6 | C | 10 | 163 | 2 | 0.0109 | 1.01 (L0.cm) | 2.61 | 0.8889 | 'Our Solar System is just one of hundreds of billions' | ' of stars in the Milky Way. The Sun, a star, and all other stars are part of the' |
| 7 | D | 3 | 170 | 3 | 0.0034 | 1.05 (L0.cm) | 2.33 | 0.929 | 'Grass is' | ' a good choice for the first few years, but it will need to be replaced more oft' |
| 8 | A | 78 | 95 | 2 | 0.0426 | 1.51 (L1.cm) | 2.29 | 0.9681 | 'So soon as Galileo had completed his elementary education, h' | ' But the young man was not a good student; he could only read and write in Latin' |
| 9 | B | 36 | 137 | 3 | 0.0136 | 0.95 (L0.cm) | 2.23 | 0.9632 | "Review and grow your students' understanding of informationa" | '\nThis product is also available in the following formats:\n- Ebook eBook\n- Digita' |
| 10 | C | 9 | 164 | 3 | 0.0040 | 1.11 (L0.cm) | 2.55 | 0.9509 | 'There is growing concern about the threats facing many' | ' species of wildlife, including polar bears.\nThe Arctic Council has been working' |
| 11 | D | 3 | 170 | 3 | 0.0167 | 1.06 (L0.cm) | 2.58 | 0.8935 | 'Birds can' | ' be trained to perform a variety of tasks, including the following:\n- Performing' |
| 12 | A | 71 | 102 | 2 | 0.0203 | 0.98 (L0.cm) | 2.21 | 0.9604 | 'Space. The material environment is a 3D space, which means a' | '\nThe 3D model can be viewed as a three-dimensional solid object that contains al' |
| 13 | B | 37 | 136 | 3 | 0.0001 | 1.07 (L0.cm) | 2.42 | 0.963 | 'Peripheral arterial disease (PAD) -- also known as periphera' | ' PAD can cause pain and discomfort when walking, leg cramps, weakness, numbness,' |
| 14 | C | 10 | 163 | 3 | 0.0038 | 1.28 (L9.tm) | 2.21 | 0.8827 | 'The Emancipation Proclamation stands as the' | ' most important document in American history. It is a declaration of freedom for' |
| 15 | D | 3 | 170 | 3 | 0.0047 | 1.76 (L12.cm) | 2.47 | 0.8639 | 'Apples are' | ' a good source of vitamin C, which is essential for the body to fight off infect' |
| 16 | A | 79 | 94 | 3 | 0.0018 | 1.67 (L4.cm) | 2.69 | 0.9677 | 'A number of fumarole communities exist on volcanic ground in' | '\nFumo (also known as fume) is a mineral compound that occurs naturally on earth,' |
| 17 | B | 37 | 136 | 3 | 0.0158 | 1.00 (L0.cm) | 2.63 | 0.9556 | 'Health promotion policy combines diverse but complementary a' | '\nThe National Health Policy (NHP) was adopted by the United Nations General Asse' |
| 18 | C | 14 | 159 | 3 | 0.0160 | 0.90 (L0.cm) | 2.22 | 0.8608 | 'This lesson covers the process of photosynthesis and the rel' | '.\n- The student will understand how to identify a leaf, flower or seed from its ' |
| 19 | D | 7 | 166 | 3 | 0.0349 | 1.09 (L18.tm) | 2.40 | 0.8606 | 'The president of the United States is' | ' elected by popular vote for a four-year term. The President is elected to a fiv' |
| 20 | A | 77 | 96 | 3 | 0.0060 | 1.35 (L7.tm) | 2.69 | 0.9368 | 'Warning is hereby given that not all Project Ideas are appro' | '\nThis page first made public: Jun 01, 2010\nLast Modified: June 02, 2013\nThe auth' |
| 21 | B | 47 | 126 | 3 | 0.0232 | 1.37 (L4.tm) | 2.82 | 0.952 | 'In 1965 Arno Penzias and Robert Wilson detected the cosmic m' | ' The first stars were born in these clouds of gas and dust that would later beco' |
| 22 | C | 11 | 162 | 4 | 0.0173 | 0.99 (L0.cm) | 2.33 | 0.9689 | 'In August 2014, US students submitted proposals for projects' | ' would have allowed them to use the data in their research.\nThe project was desi' |
| 23 | D | 3 | 170 | 3 | 0.0093 | 1.41 (L7.cm) | 2.49 | 0.8284 | 'A triangle has' | ' four sides, and the sum of its angles is 180°.\nThe sum of two adjacent angles e' |
| 24 | A | 80 | 93 | 4 | 0.0192 | 1.08 (L21.tm) | 1.89 | 0.9891 | 'Imagery is language which appeals to the five senses. It doe' | '\nThe most important thing to remember is that all of these senses are part of th' |
| 25 | B | 42 | 131 | 3 | 0.0007 | 1.14 (L1.tm) | 2.22 | 0.9308 | 'Schizophrenia is a severe mental illness that affects about ' | '\nPeople who have been diagnosed with schizophrenia are at risk for developing ot' |
| 26 | C | 13 | 160 | 4 | 0.0133 | 1.17 (L1.tm) | 2.73 | 0.9686 | 'People have used petroleum for thousands of years, for a var' | ' purposes. The earliest known use was as an oil substitute in the early 19th cen' |
| 27 | D | 6 | 167 | 4 | 0.0065 | 0.91 (L0.cm) | 3.12 | 1.0 | '1, 2, 3,' | ' 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 2' |
| 28 | A | 71 | 102 | 4 | 0.0104 | 1.78 (ln_out) | 2.88 | 0.9604 | 'In this video we learn how to understand the Rule of Vertica' | '\nThis is a great exercise for students who need help with their geometry skills,' |
| 29 | B | 34 | 139 | 3 | 0.0145 | 1.87 (L4.tm) | 2.43 | 0.9203 | 'A colloquial term for several overuse conditions resulting f' | '\n- The act or process of extending; extension; as, to extend the arm; to stretch' |
| 30 | C | 15 | 158 | 4 | 0.0214 | 1.06 (L1.cm) | 2.43 | 0.9299 | 'Kingdoms are the largest groups, with millions of different ' | ' each kingdom. Each species has its own unique characteristics and is a separate' |
| 31 | D | 4 | 169 | 4 | 0.0149 | 1.08 (L16.cm) | 2.25 | 0.9643 | 'Red, orange,' | ' yellow and green.\n- The leaves are alternate, simple, pinnate (with a few small' |
| 32 | A | 76 | 97 | 4 | 0.0016 | 1.36 (L3.tm) | 2.48 | 0.9375 | 'Slavery: cause and catalyst of the civil war us department o' | '\nThe Civil War was an important event in the history and culture of the united s' |
| 33 | B | 42 | 131 | 3 | 0.0103 | 1.56 (L5.tm) | 2.21 | 0.9154 | 'Fossils dating from before the Pleistocene glaciations show ' | '\nThe fossil record shows that the first members of this group appeared in Europe' |
| 34 | C | 13 | 160 | 4 | 0.0119 | 1.32 (L20.tm) | 2.78 | 0.9057 | 'Largely desert with some limited potential for urban and sed' | ', the area is now a popular tourist destination.\nThe site of the ancient city wa' |
| 35 | D | 3 | 170 | 4 | 0.0082 | 1.61 (ln_out) | 2.71 | 0.9349 | 'The Pacific Ocean' | ' is the largest ocean basin in the world, and it contains about half of all the ' |
| 36 | A | 72 | 101 | 4 | 0.0198 | 1.48 (L6.tm) | 2.24 | 0.93 | 'Frederick Douglass (born Frederick Augustus Washington Baile' | '\nBorn in New York City, Douglass moved to New York City at age 14, where he was ' |
| 37 | B | 39 | 134 | 4 | 0.0039 | 1.03 (L0.cm) | 2.41 | 0.9549 | "They bonded with other student activists. Berkeley's activis" | '\nThe first major anti-war movement in the United States, the Communist Party of ' |
| 38 | C | 11 | 162 | 4 | 0.0082 | 1.00 (L0.cm) | 2.39 | 0.8944 | 'Since climate change affects everyone on Earth, scientists h' | ' trying to understand how it will affect the environment.\nThe study of global wa' |
| 39 | D | 7 | 166 | 4 | 0.0082 | 1.61 (ln_out) | 2.71 | 0.9333 | 'The Pacific Ocean is the largest ocean' | ' basin in the world, and it contains about half of all the water on earth. It ha' |
| 40 | A | 80 | 93 | 4 | 0.0039 | 1.89 (L4.tm) | 2.45 | 0.9239 | 'The Moche (or Mochica) were a civilisation who occupied the ' | '\nThe Mochica are believed to have been descended from the ancient inhabitants of' |
| 41 | B | 40 | 133 | 4 | 0.0072 | 0.96 (ln_out) | 2.51 | 0.9545 | 'The Southern Ocean has cooled to?1.8C over the past 30 milli' | '\n- The Southern Ocean is a vast region of water that lies between the Antarctic ' |
| 42 | C | 11 | 162 | 4 | 0.0003 | 2.55 (L4.tm) | 2.47 | 0.8944 | 'Political, social and economic grievances in early twentieth' | ' Russia.\nThe first major political movement to challenge the autocracy of Nichol' |
| 43 | D | 3 | 170 | 4 | 0.0106 | 1.08 (L0.cm) | 2.64 | 0.9172 | 'The sun is' | ' the source of all life, and it is the only one that can produce energy. The sun' |
| 44 | A | 68 | 105 | 4 | 0.0054 | 2.03 (L14.cm) | 1.93 | 0.9135 | 'Following the 2009 outbreak of the H1N1 pandemic flu and the' | '\nThe CDC has been working with the National Institutes of Health (NIH), the Cent' |
| 45 | B | 33 | 140 | 4 | 0.0071 | 1.04 (L19.tm) | 2.60 | 0.9496 | 'The bulk modulus describes how a substance reacts when squee' | ' If the mass density is small, then the material will be relatively elastic and ' |
| 46 | C | 14 | 159 | 4 | 0.0067 | 1.10 (L1.tm) | 2.30 | 0.8924 | 'The channel length is defined by the diameter of the nanowir' | ' can be determined from the measurement. The width of a nanotube is measured in ' |
| 47 | D | 7 | 166 | 4 | 0.0241 | 2.02 (L22.tm) | 2.65 | 0.9091 | 'The Amazon is the largest river in' | ' South America, and it flows through Brazil. It has a basin of about 1,200 squar' |
| 48 | A | 64 | 109 | 4 | 0.0016 | 1.28 (L1.tm) | 2.17 | 0.9074 | 'Over the centuries, paper has been made from a wide variety ' | '\nThe first step in making paper is to remove all the fibers that are not necessa' |
| 49 | B | 40 | 133 | 4 | 0.0047 | 1.64 (L22.tm) | 2.27 | 0.947 | 'For many decades, the Death Valley mark was not considered t' | '\nThe record is now broken, however, and the record has been broken twice more si' |
| 50 | C | 10 | 163 | 4 | 0.0180 | 0.97 (L0.cm) | 2.46 | 0.821 | 'By applying the new data to the propagation equation of' | ' the system, we can derive a number of important parameters for the model.\nThe f' |
| 51 | D | 3 | 170 | 4 | 0.0109 | 1.71 (ln_out) | 2.56 | 0.9053 | 'The longest river' | ' in the world, the Nile is a major transportation artery for Egypt. It runs from' |
| 52 | A | 74 | 99 | 4 | 0.0104 | 1.38 (ln_out) | 2.19 | 0.898 | 'The drag coefficient is a common measure in automotive desig' | '\nThe drag coefficient for a given car can be calculated by using the following e' |
| 53 | B | 44 | 129 | 4 | 0.0087 | 1.05 (L1.tm) | 2.53 | 0.9375 | 'Several food items are thought to help in preventing brain c' | '\nThe study was conducted on mice with high levels of free radicals, which can da' |
| 54 | C | 11 | 162 | 5 | 0.0217 | 1.12 (L1.tm) | 2.19 | 0.9814 | 'Projection refers to the unconscious transference of psychic' | ' from one person to another.\nThe term "projections" is sometimes used in a broad' |
| 55 | D | 3 | 170 | 4 | 0.0010 | 1.27 (L1.tm) | 2.26 | 0.8994 | 'Dogs are' | ' not the only ones who can suffer from a heart attack.\nA study published in the ' |
| 56 | A | 71 | 102 | 4 | 0.0154 | 1.34 (L4.cm) | 2.13 | 0.8713 | 'Before the B-29 could take flight over Japan, America needed' | '\nThe first step toward this goal came with the construction of a new Air Force B' |
| 57 | B | 47 | 126 | 4 | 0.0012 | 1.00 (L0.cm) | 2.69 | 0.912 | "Excel's Accounting number format adds decimal to the value i" | '\nA formula is a set of instructions that can be used to perform specific calcula' |
| 58 | C | 9 | 164 | 5 | 0.0000 | 4.08 (L8.tm) | 2.50 | 0.9571 | 'We hope that the free math worksheets have' | ' helped you to understand how to solve these problems.\nIf you are still having t' |
| 59 | D | 7 | 166 | 4 | 0.0043 | 22.89 (ln_out) | 1.85 | 0.8727 | 'The expedition reached the summit after weeks' | ' of hard work, and found\r\n\r\nnothing but a few scattered trees.\r\n"We have been he' |
| 60 | A | 65 | 108 | 5 | 0.0171 | 1.11 (L1.tm) | 2.63 | 0.9907 | 'Many parts of Africa have been deforested by the expansion o' | '\nThe destruction of forests also affects the quality of water, which is needed t' |
| 61 | B | 45 | 128 | 5 | 0.0092 | 0.99 (L0.cm) | 2.59 | 0.9843 | 'Food poisoning causes symptoms of nausea, vomiting, stomach ' | '\nThe most common cause of foodborne illness is from eating foods that have been ' |
| 62 | C | 8 | 165 | 5 | 0.0091 | 3.15 (L16.cm) | 2.69 | 0.9451 | 'Most patients with allergies can benefit from the' | ' use of a nasal spray or an antihistamine.\nAllergy shots are used to reduce symp' |
| 63 | D | 6 | 167 | 4 | 0.0010 | 1.37 (L4.tm) | 2.42 | 0.8554 | 'The Second World War ended in' | ' 1945, and the United Nations was established. The United States became a member' |
