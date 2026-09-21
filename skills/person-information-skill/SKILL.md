---
name: Person Information Skill
description: Look up personal records, including person name, city and country where they live, or job title.
Trigger Queries:
  - Who lives in Tokyo according to the registry?
  - What is the job title of Alex Morgan?
  - Find all employees located in Germany
  - Look up the Chief Technology Officer in the database
  - Search for staff members in Singapore
---

# Person Information Skill

## Overview
This skill queries the local flat-file personnel database (`skills/person-information-skill/data/registry.csv`) containing 20 verified organizational records. It matches queries against employee names, residing cities, countries, and professional job titles.

## Standard Operating Procedure (SOP)
1. **Identify Query Criteria**: Extract target entity name, designated location (city/country), or target job title from the inquiry.
2. **Execute Registry Lookup**:
   - Query `data/registry.csv` by filtering rows where query keywords match the fields `name`, `city`, `country`, or `job_title`.
   - Use `skills/person-information-skill/scripts/person_search.py` or the internal python registry reader.
3. **Format Response**: Present matching candidates with their full name, city, country, and official role.
