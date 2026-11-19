# Results

This page gives the full numbers from the search. It holds no coordinates. Positions of every feature, including the ones that were rejected, were kept off this repository on purpose, because publishing them could put real sites at risk if any of them turned out to be real.

## Coverage

The EBA archive lists 901 transects. Each one is a strip of airborne lidar about 12.5 km long and 300 m wide. A 3 m ground model was built for 899 of them. The other two (0082 and 0865) are listed in the footprint file but have no point cloud in the Zenodo archive.

| State | Transects with a ground model | Features examined |
| --- | ---: | ---: |
| Pará | 304 | 1,368 |
| Amazonas | 286 | 855 |
| Mato Grosso | 126 | 396 |
| Rondônia | 72 | 197 |
| Roraima | 42 | 261 |
| Amapá | 25 | 51 |
| Maranhão | 23 | 311 |
| Acre | 20 | 97 |
| Tocantins | 1 | 1 |
| **Total** | **899** | **3,537** |

The downloads took about 3 hours of transfer time on the Open Math OS machine, and the processing took about 6.6 hours of compute. The rest of the 4.5 day run went into building detectors, testing them, arguing about the results and auditing the archive.

## Outcome by certainty

| Certainty | Count | Meaning |
| --- | ---: | --- |
| Known | 1 | Matches a recorded site. This is the positive control. |
| Strong | 0 | A clear earthwork shape with no record nearby and no other explanation left. |
| Possible | 42 | A regular shape that could be human-made, but a natural or modern explanation is still open. |
| Rejected | 3,494 | Explained as natural ground, a modern structure or a processing artefact, with a written reason. |

The 42 Possible features are 20 mound clusters, 16 short raised lines, 4 small rings, one open arc and one straight shallow depression. A manual look at every one of them found that most are gully heads, stream junctions, plough lines, field plots, roads or processing stripes. Six could not be explained from the images alone. They are five straight raised lines about 100 m long under 10 to 23 m of forest, and one flat surface covered with small regular bumps. Natural levees, old forest tracks and natural earth mounds can all make these shapes, and none of the six looks like a geoglyph.

Rejected features by detector type:

| Detector type | Rejected |
| --- | ---: |
| Mound cluster | 1,191 |
| Raised line | 1,168 |
| Circle | 870 |
| Causeway | 196 |
| U shape or arc | 41 |
| Irregular | 20 |
| Square or angular | 8 |

## The positive control

Transect 0414 in Acre crosses a recorded geoglyph. It is a closed circular ditch about 98 m across with about 1.3 m of relief, in cleared pasture. It appears in the national site register (CNSA), in Amazon Arch and in PAST, and it matches the register entry "Acre Circle 90m" within a few metres. Both pipelines in this repository show it clearly.

## The under-forest test that failed

Peripato et al. (2023) reported 24 earthworks found under forest in carbon-survey lidar. Five transects contain published positions of some of them. At all five, the detector could read between 0 and 0.2 percent of the ground in a 100 m window around the published point. More than half of the ground cells there had to be reconstructed because no laser pulse reached the soil. No ditch, ring or bank is visible in any of the images at those places.

This leaves two explanations that the data cannot separate. The EBA point density (about 2 points per square metre) and the ground filters used here may be unable to show earthworks under tall forest. The published positions may also be displaced on purpose, which is common for archaeological sites. Either way, the search was never shown to work under forest, so finding nothing under forest is not evidence that nothing is there.

## The long-road detector that failed

Sunken roads more than 200 m long are a common part of Acre geoglyph sites. The swarm built a test bench with synthetic ditches, held-out cases and natural lookalikes such as valleys and river terraces. The best detector reached a recall of 0.33 on the held-out cases and produced several hundred metres of false lines on valley and terrace scenes. It was never switched on for the real data.
