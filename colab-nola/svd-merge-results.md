# EuroSAT prompt merges: all ten classes

Complete linkage on full-space cosine ≥ 0.88; class-local SVD at 90% centered energy ranks pairs and flags disagreements. Projected-distance screening uses 0.489898, the full-space cutoff expressed as distance for unit vectors.

| Class | Input | Prototypes | SVD-only close pairs | SVD directions |
|---|---:|---:|---:|---:|
| Annual Crop Land | 10 | 6 | 8 | 7 |
| Forest | 10 | 8 | 1 | 7 |
| Herbaceous Vegetation Land | 10 | 9 | 4 | 7 |
| Highway or Road | 10 | 6 | 3 | 7 |
| Industrial Buildings | 10 | 5 | 4 | 7 |
| Pasture Land | 10 | 7 | 2 | 7 |
| Permanent Crop Land | 10 | 9 | 3 | 7 |
| Residential Buildings | 10 | 8 | 8 | 7 |
| River | 10 | 7 | 1 | 7 |
| Sea or Lake | 10 | 7 | 2 | 7 |

SVD-only close pairs pass the projected cutoff but not the full-space cosine cutoff. They were not merged.

## Every prototype and its source descriptions

### Annual Crop Land
- **P00** (merged; min cosine 0.907): rectangular fields with alternating green crop and brown soil strips / large uniform green fields with sharp rectangular borders
- **P01** (merged; min cosine 0.922): irregular farm parcels with contour-following planted rows / wide rectangular plots separated by narrow farm tracks
- **P03** (merged; min cosine 0.892): long parallel seed rows on exposed soil / bright young crop rows running parallel over bare earth
- **P04** (merged; min cosine 0.892): freshly tilled brown fields with straight furrows / light tan harvested fields with repeated short rows
- **P02**: patchwork of neighboring plots at different growth stages
- **P05**: harvested fields with pale stubble and parallel machinery tracks

### Forest
- **P06** (merged; min cosine 0.891): forest edge with a jagged boundary beside open land / woodland surrounding a small clearing
- **P08** (merged; min cosine 0.943): dense continuous dark-green canopy with irregular treetop texture / mixed tree canopy with small gaps and several green tones
- **P07**: wooded hills with canopy shadows changing across the slope
- **P09**: uniform evergreen canopy with fine speckled texture
- **P10**: leaf-off woodland with gray-brown textured crowns
- **P11**: unbroken dark woodland with tightly packed irregular crowns
- **P12**: rounded broadleaf crowns forming a mottled canopy
- **P13**: closely packed needleleaf trees with dark pointed crown shadows

### Herbaceous Vegetation Land
- **P17** (merged; min cosine 0.899): patchy dry grasses and herbs interspersed with bare soil / coarse tufts of grasses alternating with short sparse cover
- **P14**: open low herbaceous cover with no distinct tree crowns
- **P15**: flat open herbaceous cover lacking individual tree-crown shadows
- **P16**: low ground cover mixed with scattered light-colored stones
- **P18**: irregular open hillside covered in low grasses and forbs
- **P19**: continuous bright meadow vegetation with a smooth fine texture
- **P20**: low sedges and reeds in a damp mosaic of vegetation and wet ground
- **P21**: seasonally browned low vegetation with ragged patches of green
- **P22**: flower-rich meadow showing fine mottling rather than raised crowns

### Highway or Road
- **P27** (merged; min cosine 0.880): road running diagonally between built-up blocks / broad arterial with a visible central median / junction where multiple paved roads converge
- **P23** (merged; min cosine 0.883): broad divided motorway with two parallel carriageways / highway interchange with looping ramps and crossing lanes
- **P26** (merged; min cosine 0.952): straight paved roadway cutting across open fields / long straight paved strip crossing farmland
- **P24**: curving paved road through forest with a narrow cleared margin
- **P25**: winding mountain road tracing repeated bends
- **P28**: straight light-gray roadway bordered by narrow shoulders

### Industrial Buildings
- **P29** (merged; min cosine 0.889): logistics depots with truck bays along large rectangular buildings / large isolated processing hall beside a paved service yard / big rectangular warehouse buildings with extensive hardstanding
- **P31** (merged; min cosine 0.898): industrial plant with round tanks beside boxy buildings / factory buildings with dark roofs and broad concrete aprons
- **P32** (merged; min cosine 0.913): large flat-roofed warehouses surrounded by paved yards / multiple wide rectangular factory roofs in a regular grid
- **P33** (merged; min cosine 0.893): industrial complex with long sawtooth roofs / industrial estate with bright roofs and internal access roads
- **P30**: large sheds bordering open material storage lots

### Pasture Land
- **P34** (merged; min cosine 0.928): open grazed paddocks divided by straight fence lines / irregular grass paddocks separated by narrow tracks
- **P35** (merged; min cosine 0.907): grazing fields with patches of worn soil around access points / dry buff grazing field enclosed by vegetation boundaries
- **P36** (merged; min cosine 0.899): short-grazed grass next to a farm building or enclosure / uniform short grass fields divided by dense hedge lines
- **P37**: rolling pasture with diffuse shadows and no crop rows
- **P38**: large open paddock with a few scattered shade trees
- **P39**: fenced grassland with alternating light and dark paddocks
- **P40**: smooth green grazing field bounded by hedgerows

### Permanent Crop Land
- **P41** (merged; min cosine 0.919): vineyard with tightly spaced parallel rows and visible gaps / hillside vineyard with curved rows following contours
- **P42**: perennial tree plantation with staggered rows of individual crowns
- **P43**: leaf-off perennial rows marked by thin dark lines over soil
- **P44**: terraced perennial planting with narrow stepped plots
- **P45**: fruit orchard showing uniform tree spacing and green alleys
- **P46**: regular grid of individual fruit tree crowns across a field
- **P47**: orchard with widely spaced rounded crowns in a rectangular grid
- **P48**: small orchard blocks with several row orientations
- **P49**: olive grove with repeated dark crowns over pale dry soil

### Residential Buildings
- **P53** (merged; min cosine 0.913): suburban cul-de-sacs with spaced houses and lawns / individual homes and private yards on winding suburban roads
- **P54** (merged; min cosine 0.906): compact village houses with irregular narrow lanes / ribbon settlement of houses along a single street
- **P50**: rectangular urban blocks of homes on an orthogonal street grid
- **P51**: low-rise homes with bright roofs on small fenced plots
- **P52**: closely packed rows of terraced houses with repeated narrow roofs
- **P55**: detached houses with small gardens along curving streets
- **P56**: apartment blocks around open communal courtyards
- **P57**: residential towers with long shadows and lawn between buildings

### River
- **P62** (merged; min cosine 0.886): straightened watercourse between parallel banks / river junction where two channels merge / river crossing a built-up area between engineered banks
- **P61** (merged; min cosine 0.905): narrow winding water channel bordered by strips of trees / sinuous narrow waterway with wooded banks
- **P58**: wide meandering river with pale inner-bend sandbars
- **P59**: dark sinuous channel with a visible oxbow nearby
- **P60**: shallow bright channel with patches of exposed bed
- **P63**: muddy river reach winding through a flat floodplain
- **P64**: braided river channels separated by exposed gravel bars

### Sea or Lake
- **P65** (merged; min cosine 0.881): coastal water with a pale shallows band near a beach / sea meeting a straight sandy coastline
- **P67** (merged; min cosine 0.928): large lake with wooded irregular shores and coves / lake surrounding a small island or peninsula
- **P69** (merged; min cosine 0.957): broad open deep-blue water with no visible shoreline / wide uninterrupted dark-blue open water
- **P66**: broad flat turquoise coastal water with no narrow channel
- **P68**: large lake edge with irregular shallow green margins
- **P70**: large open water with wind-streaked textured surface
- **P71**: smooth dark inland water bordered by rocky shoreline

## Closest SVD-only pairs (not merged)
- **Residential Buildings**: ribbon settlement of houses along a single street ↔ individual homes and private yards on winding suburban roads (SVD distance 0.392; full cosine 0.879).
- **Permanent Crop Land**: perennial tree plantation with staggered rows of individual crowns ↔ regular grid of individual fruit tree crowns across a field (SVD distance 0.394; full cosine 0.859).
- **Permanent Crop Land**: perennial tree plantation with staggered rows of individual crowns ↔ leaf-off perennial rows marked by thin dark lines over soil (SVD distance 0.401; full cosine 0.861).
- **Residential Buildings**: compact village houses with irregular narrow lanes ↔ individual homes and private yards on winding suburban roads (SVD distance 0.412; full cosine 0.869).
- **Herbaceous Vegetation Land**: open low herbaceous cover with no distinct tree crowns ↔ flat open herbaceous cover lacking individual tree-crown shadows (SVD distance 0.423; full cosine 0.855).
- **Forest**: woodland surrounding a small clearing ↔ unbroken dark woodland with tightly packed irregular crowns (SVD distance 0.425; full cosine 0.858).
- **Herbaceous Vegetation Land**: open low herbaceous cover with no distinct tree crowns ↔ seasonally browned low vegetation with ragged patches of green (SVD distance 0.431; full cosine 0.834).
- **Herbaceous Vegetation Land**: patchy dry grasses and herbs interspersed with bare soil ↔ continuous bright meadow vegetation with a smooth fine texture (SVD distance 0.434; full cosine 0.854).
- **Herbaceous Vegetation Land**: patchy dry grasses and herbs interspersed with bare soil ↔ irregular open hillside covered in low grasses and forbs (SVD distance 0.449; full cosine 0.858).
- **Pasture Land**: open grazed paddocks divided by straight fence lines ↔ dry buff grazing field enclosed by vegetation boundaries (SVD distance 0.451; full cosine 0.876).
