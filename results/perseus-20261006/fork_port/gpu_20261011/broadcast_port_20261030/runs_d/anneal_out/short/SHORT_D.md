# Summarization eval, before and after the anneal (gen_summ.py)

Quality only (R4). Every row comes from the JSONL named beside it. Words = whitespace-free a-z0-9 runs.

## Held-out BillSum (US test pairs that fit in 2,048 tokens)

| model | weights sha256 | policy | n | rougeL_f1_own | unigram_f1 | rouge1_f1 | rouge2_f1 | rougeL_f1 | mean words |
|---|---|---|---|---|---|---|---|---|---|

Sources: `short_pbd430a.jsonl`, `short_pbd430s.jsonl`


## Document D, all cue × policy combinations

Judge each against the 8 facts and the "wrong" list in `D_reference.md`. Every output is shown, not the best.

### pbd430a (weights sha256 3ec3f1d8b9b71e7eee38055a7621213991c6417a89576f1c02590963293e784b)

**cue "\n\nTL;DR:", norepeat** (90 tokens, eos False, 80 words)

>  Tallying accounts is a simple way to keep track of your spending. You can use it for all expenses (including rent) as well as for items that you have purchased or used in the past year. It also allows you to create an account with a specific purpose, such as checking out a new car or buying a house.
> 1. The amount of money you spend on your bill is determined by the number of days you use it each

**cue "\n\nTL;DR:", greedy** (90 tokens, eos False, 72 words)

>  Tallying is a process of checking the balance of your account, and then checking the balance of the account.
> Tallying is a simple process, and you can do it in a few minutes.
> 1. First, you need to know the amount of money you have in your account.
> 2. You need to know the amount of money you have in your account.
> 3. You need to know how much you have in

**cue "\n\nIn plain English, these terms mean:", norepeat** (90 tokens, eos False, 72 words)

> 
> 1. The service will be free for you, but we do not charge a fee;
> 2. You can choose to pay only when you want to or if you prefer it;
> 3. We have no obligation to return the money back to you;
> 4. If you decide to cancel your account, you may withdraw from our service at any time;
> 5. The payment is not subject to a penalty for non-payment of

**cue "\n\nIn plain English, these terms mean:", greedy** (90 tokens, eos False, 76 words)

> 
> 1. The terms of the contract are not to be changed without the consent of the other party.
> 2. The terms of the contract are not to be changed without the consent of the other party.
> 3. The terms of the contract are not to be changed without the consent of the other party.
> 4. The terms of the contract must be written in the contract.
> 5. The terms of the contract must be written in

**cue "\n\nSummary:", norepeat** (90 tokens, eos False, 76 words)

>  Tallying accounts is a simple process that can be used for many purposes. It provides an easy way to track your expenses, as well as provide a means of tracking your spending over time. You may use the program to keep track of all your expenditures, including those made on your own or with others.
> The software works by recording and displaying the amount you spend each month in various categories: food, shelter, transportation, utilities, etc.,

**cue "\n\nSummary:", greedy** (90 tokens, eos False, 79 words)

>  Tallying is a process of determining the value of a service or product, and then comparing the value of the value to the cost of the service.
> The value of a service is the value of the value of the service.
> The value of a service is the value of the value of the service.
> The value of the service is the value of the value of the service.
> The value of the service is the value of the

**cue "\n\nIn short, by signing up for Tallyhen you agree that", norepeat** (90 tokens, eos False, 82 words)

>  you will not use your account as a means of making money, but rather as an alternative to paying bills on time or with insufficient funds. You can also opt out of the program at any time, if you choose to do so.
> 6. The amount paid is based upon the number of accounts you have used in the past year, and the total amount of your account balance will be calculated according to the current monthly payment schedule. If you are

**cue "\n\nIn short, by signing up for Tallyhen you agree that", greedy** (90 tokens, eos False, 73 words)

>  you will not use the service for any purpose other than the purpose for which it was created.
> If you have any questions, please contact us.
> This page was last modified on:
> 21 May 2013
> Page last modified:
> 21 May 2013
> This page was last modified on
> 21 May 2013
> This page was last modified on
> 21 May 2013
> This page was last modified on
> 21 May 2013
> This page was last modified

### pbd430s (weights sha256 766d81789f67ec34795f715b52f97b4e3da494b4d4a01333355569f4eeb8018a)

**cue "\n\nTL;DR:", norepeat** (90 tokens, eos False, 75 words)

>  Tellingtale has a reputation for being a reliable service provider. We have been providing quality customer support since we launched in 2016. Our customers love our experience and offer a wide range of services to meet your needs.
> Our team is dedicated to meeting the demands of our clients, ensuring that they get what they need while maintaining their business operations. Tellingtail has an excellent reputation for delivering high-quality service at affordable prices. We are

**cue "\n\nTL;DR:", greedy** (90 tokens, eos False, 65 words)

>  Tellingtale is a free, open source, open source, open source, open source, free, and open source software platform that is designed to be used as a platform for a wide range of applications.
> It is a free, open source, open source, open source, open source, open source, open source, open source, open source, open source, open source, open source, open source, open source, open source

**cue "\n\nIn plain English, these terms mean:", norepeat** (90 tokens, eos False, 77 words)

> 
> 1. A service that allows you to pay for your phone bill from anywhere in the world without having to make a call to your local telephone company or exchange carrier.
> 2. The ability to access your account via an Internet connection.
> 3. The ability of anyone who owns a mobile device to connect to it and use it as a terminal, computer, or other device.
> 4. The ability for you to pay the bill on any

**cue "\n\nIn plain English, these terms mean:", greedy** (90 tokens, eos False, 67 words)

> 
> - A business that is not a part of the Telling Telling program.
> - A business that is not a Telling business.
> - A business that is not a Telling business.
> - A business that is not a Telling business.
> - A business that is not a Telling business.
> - A business that is not a Telling business.
> - A business that is not a Telling business.
> 

**cue "\n\nSummary:", norepeat** (90 tokens, eos False, 65 words)

>  Telling Time with Tallypak
> Talentap provides a simple way for people to share their time on social media without having to wait for an online service or phone call. It is free of charge, easy-to-use, and offers the best value when compared to other apps that offer similar features.
> How To Use Tallypang:
> Tingalap provides a secure way to store your accounts and manage them

**cue "\n\nSummary:", greedy** (90 tokens, eos False, 71 words)

>  Telling Time with Tally is a free, online, and mobile app for the free and open sharing of information about your life, your finances, and your finances.
> Tally is a free app that allows you to create and share your own accounts. It is free for Android and iOS devices.
> Tally is free for Android and iOS devices.
> Tally is free for Android and iOS.
> Tally is free for Android and

**cue "\n\nIn short, by signing up for Tallyhen you agree that", norepeat** (90 tokens, eos False, 74 words)

>  you will not sell your data to anyone else. You also agree to pay a reasonable fee per month (up to $100) to keep the service running.
> If you want to share your data with other people or companies, you can do so using our online services.
> You may choose to use Tallying’s own software for this purpose. We offer free access to our database and allow users to create their own databases.
> We

**cue "\n\nIn short, by signing up for Tallyhen you agree that", greedy** (90 tokens, eos False, 69 words)

>  you will not sell, resell or transfer your property to anyone else.
> 6. How much does Tallying cost?
> Tallying is a simple process. You will need to pay the costs of the registration, the registration, the registration, the registration fee, and the fees for the processing of the application.
> The fees are collected from the seller of the property. The seller may also collect the fees from the seller.
> 

