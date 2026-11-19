# Amazon-Earthworks-Lidar

A search of 899 public lidar strips over the Brazilian Amazon for unrecorded pre-Columbian earthworks. It found nothing new, and this repository explains why.

## The result in short

We built ground models for 899 of the 901 airborne lidar strips that Brazil's EBA project flew to measure forest carbon. We screened them for ring ditches, geoglyphs, mound villages and causeways, and checked every feature against 961 recorded earthworks and the published literature. The search found **no new archaeological site**. It found one recorded geoglyph, which we used as a test, and it left 42 weak features that most likely have natural or modern causes.

The more important result is a limit of the method. At five places where earlier researchers had reported earthworks under the forest, our ground models showed nothing, because almost no laser pulses reached the soil there. That means a blank result under tall forest says nothing about what is really there. We did not solve this, and we say so plainly.

The full numbers are in [docs/results.md](docs/results.md), and a review of the automated run is in [docs/swarm_review.md](docs/swarm_review.md).

## Where the idea came from

The project was inspired by Pietro Miotti's post [*My AI Agents (Re)discovered a Maya Site in 72 Hours*](https://pietromiotti.com/latest/maya-site/). He gave a team of AI agents public NASA lidar over the Maya lowlands, and they spotted what looked like an unrecorded settlement with a plaza and a pyramid. He reported it to INAH, Mexico's heritage institute. He later learned that Estrada-Belli et al. (2025) had already mapped the site, because his agents had missed that paper in their literature check. He was honest about this in the post, and that honesty is the reason we treated the record check as the most important step here.

We tried the same idea in the Amazon. The starting point was the study by Peripato et al. (2023), [*More than 10,000 pre-Columbian earthworks are still hidden throughout Amazonia*](https://doi.org/10.1126/science.ade2541) (Science 382:103). Its authors searched about 5,300 km² of the same kind of carbon-survey lidar and found 24 earthworks that were not on record. They estimated that more than 10,000 remain hidden, mostly in the southwest Amazon.

## How the search worked

The work was done in two stages.

In the first stage, a local pipeline in [`src/`](src/) processed 12 strips in Acre by hand. It downloads one strip at a time, builds a 3 m ground model, renders hillshade and local relief images, and checks each feature against known sites. It found the test geoglyph in strip 0414 and nothing else.

In the second stage, a swarm of AI agents on [Open Math OS](https://os.openmathmodel.org) ran the full search for about four and a half days, from 13 to 17 November 2026. The agents built ground models for all available strips, ran detectors for rings, mound clusters and raised lines, argued against every detection, and checked survivors against the Peripato table, an independent KML register and the 2026 survey by Pärssinen et al. Their scripts are in [`swarm_code/`](swarm_code/).

Some lessons from the work are useful to anyone who tries this:

- The ground classification in the EBA files is wrong under forest. Points marked as ground reach up to 40 m into the canopy, so a model built from them is pure noise. Both pipelines ignore the label and find the ground themselves.
- The point density is only about 2 per m², so 1 m cells are too fine under canopy and 3 m cells work better.
- Holes in the ground data, such as ponds, buildings or dense canopy, create ring-shaped halos in relief images that look like small enclosures. The relief has to be blanked near every hole.
- A test site is essential. Without one, a pipeline can report nothing and look correct when it is blind.

## What is in this repository

| Path | Contents |
| --- | --- |
| `src/` | The local pipeline: ground model and relief tiles (`dtm.py`), resumable batch downloader (`batch.py`), region viewer with canopy height (`inspect_region.py`), and distance checks against known sites (`crosscheck.py`, `known_in_grid.py`) |
| `swarm_code/` | The scripts written by the Open Math OS swarm for processing, screening, skeptic review, record checks and archive audits |
| `docs/results.md` | All numbers from the search, with no coordinates |
| `docs/swarm_review.md` | A review of what the swarm did well and where it fell short |

Positions of features are not in this repository. Publishing exact positions of possible archaeological sites can lead to looting, so they stay private. The swarm's reference data folder was left out for the same reason, so a few audit scripts in `swarm_code/` point at files that are not included here.

## Running the local pipeline

The local pipeline needs Python 3.10 or later.

```bash
pip install -r requirements.txt
python src/batch.py 0414
python src/known_in_grid.py data/raw/NP_T-0414.laz outputs/private/T-0414
```

The first command downloads strip 0414 from Zenodo (about 220 MB) and writes the ground model and review tiles to `outputs/private/T-0414`. The second command lists recorded earthworks near the strip, and it needs the Peripato table saved as `data/ref/earthworks_known.csv` with the columns Longitude, Latitude and Database.

## Ideas for other uses

The same approach works wherever someone needs to find small shapes in the ground that trees or grass hide. These are some ideas for people who want to reuse it.

- **Archaeology in other forests.** Public lidar exists for parts of Central America, Southeast Asia, Central Africa and European woodland. Old field walls, charcoal hearths, hillforts and lost roads show up in the same kind of relief images.
- **Historic mining and industry.** Abandoned mine shafts, spoil heaps, old canals and mill ponds are often hidden in woodland. Finding them helps with land safety and with local history.
- **Wartime landscapes.** Trenches, shell craters and bunkers from the World Wars survive under forest in France, Belgium and Eastern Europe. Mapping them helps historians and also bomb disposal teams.
- **Geology and hazards.** Fault scarps, old landslides and sinkholes look like ditches and mounds in a relief model. The same filters can screen for them before a road or building is planned.
- **Ecology.** Termite mounds, beaver dams, animal trails and old river channels shape forests. Counting them over large areas is slow on foot and quick in lidar.
- **Checking a pipeline before trusting it.** The hardest lesson here applies to any automated search. Put a known example into the data first and make sure the method can find it. Then test where it cannot see, and report that limit as clearly as the results.
- **Agent swarms for slow, careful science.** The swarm spent most of its time testing detectors and correcting its own claims, which is the right use of many agents. The same setup could screen satellite images, telescope surveys or old maps for things that need checking against records before anyone calls them new.

## Data and credits

- Lidar: EBA (Estimativa de Biomassa na Amazônia) L1A transects from INPE, CC BY 4.0, on Zenodo (records 7689909, 7636454, 7689693 and 7689210), with footprints in record 4968706.
- Known sites: the data release of Peripato et al. 2023 (Zenodo record 8083287), which combines Amazon Arch, PAST, CNSA (IPHAN), INRAP and DAC, and the authors' own finds.
- Inspiration: Pietro Miotti, [*My AI Agents (Re)discovered a Maya Site in 72 Hours*](https://pietromiotti.com/latest/maya-site/).

No lidar data is stored in this repository. The code is under the MIT licence.
