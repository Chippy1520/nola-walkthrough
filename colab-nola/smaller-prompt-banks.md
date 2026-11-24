# Smaller EuroSAT prompt banks

Same 100 descriptions and CLIP vectors. Complete linkage in the original normalized CLIP space; SVD does not override the cosine cutoff. The 0.88 bank is preserved.

| Class | 0.88 baseline | 0.85 moderate | 0.80 aggressive |
|---|---:|---:|---:|
| Annual Crop Land | 6 | 5 | 4 |
| Forest | 8 | 8 | 6 |
| Herbaceous Vegetation Land | 9 | 7 | 5 |
| Highway or Road | 6 | 4 | 3 |
| Industrial Buildings | 5 | 4 | 2 |
| Pasture Land | 7 | 5 | 3 |
| Permanent Crop Land | 9 | 7 | 4 |
| Residential Buildings | 8 | 5 | 2 |
| River | 7 | 6 | 3 |
| Sea or Lake | 7 | 6 | 4 |
| Total | 72 | 57 | 36 |

## New groups at 0.85 relative to previous cutoff
- Annual Crop Land (minimum cosine 0.877): irregular farm parcels with contour-following planted rows / patchwork of neighboring plots at different growth stages / wide rectangular plots separated by narrow farm tracks
- Herbaceous Vegetation Land (minimum cosine 0.855): open low herbaceous cover with no distinct tree crowns / flat open herbaceous cover lacking individual tree-crown shadows
- Herbaceous Vegetation Land (minimum cosine 0.854): patchy dry grasses and herbs interspersed with bare soil / irregular open hillside covered in low grasses and forbs / coarse tufts of grasses alternating with short sparse cover
- Highway or Road (minimum cosine 0.859): curving paved road through forest with a narrow cleared margin / winding mountain road tracing repeated bends
- Highway or Road (minimum cosine 0.858): road running diagonally between built-up blocks / straight light-gray roadway bordered by narrow shoulders / broad arterial with a visible central median / junction where multiple paved roads converge
- Industrial Buildings (minimum cosine 0.876): logistics depots with truck bays along large rectangular buildings / large sheds bordering open material storage lots / large isolated processing hall beside a paved service yard / big rectangular warehouse buildings with extensive hardstanding
- Pasture Land (minimum cosine 0.862): open grazed paddocks divided by straight fence lines / grazing fields with patches of worn soil around access points / irregular grass paddocks separated by narrow tracks / short-grazed grass next to a farm building or enclosure / dry buff grazing field enclosed by vegetation boundaries / uniform short grass fields divided by dense hedge lines
- Permanent Crop Land (minimum cosine 0.861): perennial tree plantation with staggered rows of individual crowns / leaf-off perennial rows marked by thin dark lines over soil
- Permanent Crop Land (minimum cosine 0.874): fruit orchard showing uniform tree spacing and green alleys / regular grid of individual fruit tree crowns across a field
- Residential Buildings (minimum cosine 0.856): closely packed rows of terraced houses with repeated narrow roofs / rectangular urban blocks of homes on an orthogonal street grid / low-rise homes with bright roofs on small fenced plots
- Residential Buildings (minimum cosine 0.879): detached houses with small gardens along curving streets / compact village houses with irregular narrow lanes / ribbon settlement of houses along a single street
- River (minimum cosine 0.867): wide meandering river with pale inner-bend sandbars / dark sinuous channel with a visible oxbow nearby
- Sea or Lake (minimum cosine 0.872): large lake with wooded irregular shores and coves / lake surrounding a small island or peninsula / large lake edge with irregular shallow green margins

## New groups at 0.80 relative to previous cutoff
- Annual Crop Land (minimum cosine 0.846): long parallel seed rows on exposed soil / freshly tilled brown fields with straight furrows / light tan harvested fields with repeated short rows / bright young crop rows running parallel over bare earth
- Forest (minimum cosine 0.832): dense continuous dark-green canopy with irregular treetop texture / mixed tree canopy with small gaps and several green tones / uniform evergreen canopy with fine speckled texture
- Forest (minimum cosine 0.835): leaf-off woodland with gray-brown textured crowns / unbroken dark woodland with tightly packed irregular crowns
- Herbaceous Vegetation Land (minimum cosine 0.834): patchy dry grasses and herbs interspersed with bare soil / continuous bright meadow vegetation with a smooth fine texture / irregular open hillside covered in low grasses and forbs / coarse tufts of grasses alternating with short sparse cover
- Herbaceous Vegetation Land (minimum cosine 0.824): low sedges and reeds in a damp mosaic of vegetation and wet ground / seasonally browned low vegetation with ragged patches of green
- Highway or Road (minimum cosine 0.845): straight paved roadway cutting across open fields / road running diagonally between built-up blocks / straight light-gray roadway bordered by narrow shoulders / broad arterial with a visible central median / junction where multiple paved roads converge / long straight paved strip crossing farmland
- Industrial Buildings (minimum cosine 0.818): large flat-roofed warehouses surrounded by paved yards / multiple wide rectangular factory roofs in a regular grid / industrial complex with long sawtooth roofs / industrial plant with round tanks beside boxy buildings / factory buildings with dark roofs and broad concrete aprons / industrial estate with bright roofs and internal access roads
- Pasture Land (minimum cosine 0.827): open grazed paddocks divided by straight fence lines / grazing fields with patches of worn soil around access points / irregular grass paddocks separated by narrow tracks / rolling pasture with diffuse shadows and no crop rows / short-grazed grass next to a farm building or enclosure / large open paddock with a few scattered shade trees / dry buff grazing field enclosed by vegetation boundaries / uniform short grass fields divided by dense hedge lines
- Permanent Crop Land (minimum cosine 0.802): terraced perennial planting with narrow stepped plots / perennial tree plantation with staggered rows of individual crowns / leaf-off perennial rows marked by thin dark lines over soil
- Permanent Crop Land (minimum cosine 0.815): orchard with widely spaced rounded crowns in a rectangular grid / fruit orchard showing uniform tree spacing and green alleys / small orchard blocks with several row orientations / regular grid of individual fruit tree crowns across a field
- Residential Buildings (minimum cosine 0.803): detached houses with small gardens along curving streets / closely packed rows of terraced houses with repeated narrow roofs / apartment blocks around open communal courtyards / suburban cul-de-sacs with spaced houses and lawns / compact village houses with irregular narrow lanes / rectangular urban blocks of homes on an orthogonal street grid / low-rise homes with bright roofs on small fenced plots / ribbon settlement of houses along a single street / individual homes and private yards on winding suburban roads
- River (minimum cosine 0.801): wide meandering river with pale inner-bend sandbars / dark sinuous channel with a visible oxbow nearby / shallow bright channel with patches of exposed bed
- River (minimum cosine 0.807): narrow winding water channel bordered by strips of trees / straightened watercourse between parallel banks / river junction where two channels merge / muddy river reach winding through a flat floodplain / river crossing a built-up area between engineered banks / sinuous narrow waterway with wooded banks
- Sea or Lake (minimum cosine 0.817): coastal water with a pale shallows band near a beach / broad flat turquoise coastal water with no narrow channel / sea meeting a straight sandy coastline
- Sea or Lake (minimum cosine 0.846): broad open deep-blue water with no visible shoreline / large open water with wind-streaked textured surface / wide uninterrupted dark-blue open water

No image predictions or training were run. These configurations are candidates, not tuned thresholds.
