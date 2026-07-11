# Recurring Meeting Notes (Cohort Standups and Onboarding)

---

## Standup - Monday

- Two new mentees still cannot push to the class repo. Same Git problem as last week: they cloned over HTTPS and never set up an SSH key, so `git push` asks for a password and fails.
- Reminder: run the daily standup in the format "yesterday / today / blockers". A few mentees are writing paragraphs instead of the three-line format.
- Deploying to the staging server keeps confusing people. You must run `make build` before `make deploy`, and only deploy from the `main` branch.
- FAQ that came up again: "How do I reset my local branch if I made a mess?" Answer: `git reset --hard origin/main` after committing anything you want to keep.

---

## Onboarding Session - Wednesday

- Walked the new batch through SSH key setup for Git again. This is the third cohort in a row where SSH keys are the first-day blocker. Steps: generate a key with `ssh-keygen`, copy the public key, add it to the account, test with `ssh -T git@host`.
- Explained the standup format once more: three lines, yesterday / today / blockers, keep it short so the whole team fits in ten minutes.
- Showed how staging deploys work: build first, then deploy, always from `main`. Someone deployed a feature branch by accident and broke the demo.
- Common question: "What do I do when the deploy fails halfway?" Answer: check the build log, fix the error, rebuild, then redeploy. Never edit files directly on the server.

---

## Standup - Friday

- Git push failures again for one mentee: wrong SSH key permissions. `chmod 600` on the private key fixed it.
- Standup format is finally sticking for most people. Keep reminding the two who still ramble.
- Staging deploy checklist worked well this time: everyone built before deploying and stayed on `main`.
- New recurring question: "How do I roll back a bad staging deploy?" Answer: redeploy the previous known-good commit from `main`.
