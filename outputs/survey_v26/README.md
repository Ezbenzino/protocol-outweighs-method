# Literature survey record (manuscript v26, 2026-10-15)

Reporting practice in LIDC-IDRI / LUNA16 lung nodule segmentation studies (section 3.7 and Supplementary Note S3).

| File | Content |
|---|---|
| `epmc_raw.json` | Europe PMC response metadata (query, retrieval time, all 164 records; abstracts removed) |
| `decisions.json` | screening decision and reason for every record (keyed by record number = position in `epmc_raw.json`) |
| `codes.json` | coded fields for the 46 full-text studies, with verbatim supporting quotations |
| `S2a_survey_records.csv` | all 164 records with bibliographic data, decision and reason |
| `S2b_survey_coding.csv` | the 46 coded studies: F/T/C/E/V codes, details and quotations |
| `Supplementary_data_S2_literature_survey.xlsx` | the two tables above plus a README sheet (submitted as supplementary data) |
| `stats.json` | counts and proportions quoted in the manuscript |
| `make_survey_outputs.py` | regenerates the CSVs, the spreadsheet and stats.json from the JSON files |
| `extract.py`, `addcode.py` | helper scripts: sentence extraction from full text, and quotation check (every quotation must occur verbatim in the full text) |

Query (Europe PMC REST, `https://www.ebi.ac.uk/europepmc/webservices/rest/search`):

    TITLE_ABS:("LIDC-IDRI" OR "LIDC" OR "LUNA16" OR "LUNA") AND TITLE_ABS:(segment*) AND TITLE_ABS:(nodule*) AND PUB_YEAR:[2018 TO 2026]

Full texts were taken from Europe PMC (`/{PMCID}/fullTextXML`) or the NCBI PMC rendering; they are not redistributed here and can be re-downloaded with the PMCIDs in `S2a_survey_records.csv`.
Coding: single rater; every coded field is accompanied by the passage on which it rests.
