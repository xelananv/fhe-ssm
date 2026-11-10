# Summarization eval, before and after the anneal (gen_summ.py)

Quality only (R4). Every row comes from the JSONL named beside it. Words = whitespace-free a-z0-9 runs.

## Held-out BillSum (US test pairs that fit in 2,048 tokens)

| model | weights sha256 | policy | n | rougeL_f1_own | unigram_f1 | rouge1_f1 | rouge2_f1 | rougeL_f1 | mean words |
|---|---|---|---|---|---|---|---|---|---|
| pbd430a | 3ec3f1d8b9b7 | norepeat | 200 | 0.1475 | 0.2409 | 0.2569 | 0.0311 | 0.1528 | 70.5 |
| pbd430a | 3ec3f1d8b9b7 | greedy | 200 | 0.1595 | 0.2211 | 0.2320 | 0.0325 | 0.1637 | 70.7 |
| lead-60 baseline | — | — | 200 | 0.2151 | 0.3161 | 0.3333 | 0.1515 | 0.2220 | 60 |
| pbd430s | 766d81789f67 | norepeat | 200 | 0.1893 | 0.3092 | 0.3314 | 0.0883 | 0.1986 | 74.4 |
| pbd430s | 766d81789f67 | greedy | 200 | 0.2184 | 0.3061 | 0.3241 | 0.1168 | 0.2261 | 64.0 |

Sources: `summ_pbd430a.jsonl`, `summ_pbd430s.jsonl`

- pbd430a: incremental decode vs full forward, relErrRms 6.65e-07 over 16 tokens.
- pbd430s: incremental decode vs full forward, relErrRms 6.39e-07 over 16 tokens.

## Document D, all cue × policy combinations

Judge each against the 8 facts and the "wrong" list in `D_reference.md`. Every output is shown, not the best.

### pbd430a (weights sha256 3ec3f1d8b9b71e7eee38055a7621213991c6417a89576f1c02590963293e784b)

**cue "\n\nTL;DR:", norepeat** (90 tokens, eos False, 72 words)

>  We are not able to provide legal advice on any matter related to this topic. Please refer to our Legal Information page for more information.
> Last updated on: 12/08/2014
> © 2014 LawLectures LLC | All Rights Reserved
> This document can be downloaded in its entirety, or part of a document, as an attachment from the web site. You may not distribute it without written permission.
> Please note that this file contains copyrighted

**cue "\n\nTL;DR:", greedy** (90 tokens, eos False, 79 words)

>  We are not able to provide you with a copy of the original contract, but we will be happy to provide you with a copy of the original contract.
> We will be happy to provide you with a copy of the original contract.
> We will not accept any other terms than those listed below.
> We will not accept any other terms than those listed below.
> We will not accept any other terms than those listed below.
> We will not

**cue "\n\nIn plain English, these terms mean:", norepeat** (90 tokens, eos False, 75 words)

> 
> 1. We are not responsible for any actions taken by you that may affect your rights under this Act.
> 2. You have no control over the use of our services or products.
> 3. If you decide to change your mind about a matter, we will make it clear and we shall do so in writing.
> 4. We are willing to work with you to resolve any issues that may arise.
> 5. We will not take

**cue "\n\nIn plain English, these terms mean:", greedy** (90 tokens, eos False, 72 words)

> 
> - "We are not responsible for the content of any comments, questions, or comments you make on this page.
> - You may not post any personal information, including your name, address, phone number, email address, or any other personal information, on this site.
> - We do not store personal information in any form.
> - We do not store any personal information in any form.
> - We do not store any personal information

**cue "\n\nSummary:", norepeat** (90 tokens, eos False, 84 words)

>  The purpose of this document is to provide a brief overview of the legal issues that arise when dealing with the issue of a contract between two parties who are not related to each other and whose relationship may be affected by the terms of the agreement. It does so in order to give you an idea of what the terms mean, but it cannot replace the advice of a lawyer or legal adviser.
> This document is available as a PDF file (PDF) for

**cue "\n\nSummary:", greedy** (90 tokens, eos False, 81 words)

>  The purpose of this document is to provide information about the legal status of the property, the legal status of the parties, and the legal status of the parties.
> The purpose of this document is to provide information about the legal status of the property, and to provide information about the legal status of the parties.
> The information contained in this document is provided for educational and informational purposes only. It is not intended to be a legal document, and it

**cue "\n\nIn short, by signing up for Tallyhen you agree that", norepeat** (90 tokens, eos False, 83 words)

>  you will not use any of our services to make your life easier, but we do ask that you respect our privacy and the rights of others before making a purchase or using this service. We also promise never to sell, lease, or give away your personal information in any way without your written consent.
> We are committed to providing you with the best possible service, and we hope you will consider us when you make decisions about how to use our services

**cue "\n\nIn short, by signing up for Tallyhen you agree that", greedy** (90 tokens, eos False, 76 words)

>  you will not use the information provided to you to make any decisions about your health or medical care.
> We will not sell, lease, or sell your personal information to any third party.
> We will not sell your personal information to any third party.
> We will not sell your personal information to third parties.
> We will not sell your personal information to third parties.
> We will not sell your personal information to third parties.
> We will

### pbd430s (weights sha256 766d81789f67ec34795f715b52f97b4e3da494b4d4a01333355569f4eeb8018a)

**cue "\n\nTL;DR:", norepeat** (90 tokens, eos False, 83 words)

>  The information provided on this page is not a substitute for legal advice and should be used only as an aid to decision making. It does NOT replace professional legal advice from your own attorney. If you have any questions about the law or any other matter related to your personal situation, please consult with a qualified professional.
> This site may contain links that we believe are useful, but we cannot guarantee their accuracy and completeness. We do our best to make sure

**cue "\n\nTL;DR:", greedy** (90 tokens, eos False, 80 words)

>  The information provided in this document is not legal advice. It is intended to provide general information and is not intended to be legal advice.
> The information provided is not intended to be a substitute for legal advice. It is not intended to be a substitute for professional legal advice.
> This page is not intended to be a substitute for professional legal advice.
> This page is not intended to be a substitute for professional legal advice.
> This page is not

**cue "\n\nIn plain English, these terms mean:", norepeat** (90 tokens, eos False, 58 words)

> 
> 1. "A" means that the person is a citizen of Newfoundland and Labrador;
> 2. "b" means that you are an employee of the Department of Justice or the Office of Public Prisons;
> 3. "c" indicates that you have been convicted of a crime, such as a felony conviction;
> 4. "e" is for "emergency";
> 5. "j" denotes your identity

**cue "\n\nIn plain English, these terms mean:", greedy** (90 tokens, eos False, 75 words)

> 
> - The person who is responsible for the operation of the business.
> - The person who is responsible for the business.
> - The person who is responsible for the business.
> - The person who is responsible for the operation.
> - The person who is responsible for the operation.
> - The person who is responsible for the operation.
> - The person who is responsible for the operation.
> - The person who is responsible for the operation

**cue "\n\nSummary:", norepeat** (90 tokens, eos False, 78 words)

>  The Law Society of Newfoundland and Labrador Act of 2017 This bill amends the Law of New Brunswick to allow a person who is not an employee of the Department of Justice (DOJ) or the Ministry of Justice to act as a legal advisor for any law firm that has been awarded a contract by the New Brunswick government. The Law Office of the Law Society of Nova Scotia, which was created in 1883, is responsible for licensing

**cue "\n\nSummary:", greedy** (90 tokens, eos False, 64 words)

>  The Law of the Sea Act of 2014 - Prohibits the use of the term "Coast Guard" in any law or regulation. Prohibits the use of the term "Coast Guard" in any law or regulation. Prohibits the use of the term "Coast Guard" in any law or regulation. Prohibits the use of the term "Coast Guard" in any law or regulation. Prohibits the use

**cue "\n\nIn short, by signing up for Tallyhen you agree that", norepeat** (90 tokens, eos False, 74 words)

>  you will not sell, resell, offer to sell, or otherwise cause any harm to anyone. You may also choose to cancel your subscription to the website if you wish to continue receiving our services and products. If you want to stop receiving these articles, please click here.
> You can unsubscribe at any time by clicking on the "unsubscribe" button in the top right corner of this page or by clicking the "Unsubscribe" link at

**cue "\n\nIn short, by signing up for Tallyhen you agree that", greedy** (90 tokens, eos False, 74 words)

>  you will not sell, resell, or otherwise use any of the products or services of the company.
> You agree to be bound by the terms of this agreement.
> You agree to be bound by the terms of this agreement.
> You agree to be bound by the terms of this agreement.
> You agree to be bound by this agreement.
> You agree to be bound by this agreement.
> You agree to be bound by this agreement.

