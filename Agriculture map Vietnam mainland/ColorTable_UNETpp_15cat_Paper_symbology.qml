<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis styleCategories="Symbology" version="3.44.10-Solothurn">
  <pipe-data-defined-properties>
    <Option type="Map">
      <Option value="" type="QString" name="name"/>
      <Option name="properties"/>
      <Option value="collection" type="QString" name="type"/>
    </Option>
  </pipe-data-defined-properties>
  <pipe>
    <provider>
      <resampling maxOversampling="2" enabled="false" zoomedInResamplingMethod="nearestNeighbour" zoomedOutResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer type="paletted" band="1" nodataColor="" alphaBand="-1" opacity="1">
      <rasterTransparency/>
      <minMaxOrigin>
        <limits>None</limits>
        <extent>WholeRaster</extent>
        <statAccuracy>Estimated</statAccuracy>
        <cumulativeCutLower>0.02</cumulativeCutLower>
        <cumulativeCutUpper>0.98</cumulativeCutUpper>
        <stdDevFactor>2</stdDevFactor>
      </minMaxOrigin>
      <colorPalette>
        <paletteEntry value="1" color="#0623df" label="Water surface" alpha="255"/>
        <paletteEntry value="2" color="#d80340" label="Building" alpha="255"/>
        <paletteEntry value="3" color="#8897d3" label="Aquaculture" alpha="255"/>
        <paletteEntry value="4" color="#f1fa29" label="Rice paddy field" alpha="255"/>
        <paletteEntry value="5" color="#988534" label="Coffee" alpha="255"/>
        <paletteEntry value="6" color="#0cec84" label="Grassland" alpha="255"/>
        <paletteEntry value="7" color="#ff742b" label="Orchard" alpha="255"/>
        <paletteEntry value="8" color="#0d81c3" label="Melauleca" alpha="255"/>
        <paletteEntry value="9" color="#85db0e" label="Mangrove" alpha="255"/>
        <paletteEntry value="10" color="#118c0a" label="Evergreen&#xa;Broadleaf Forest" alpha="255"/>
        <paletteEntry value="11" color="#dea41b" label="Rubber tree" alpha="255"/>
        <paletteEntry value="12" color="#ff72e0" label="Barren" alpha="255"/>
        <paletteEntry value="13" color="#a3ff72" label="Coconut" alpha="255"/>
        <paletteEntry value="14" color="#8300a8" label="Crop" alpha="255"/>
        <paletteEntry value="15" color="#a73700" label="Cashew" alpha="255"/>
      </colorPalette>
      <colorramp type="randomcolors" name="[source]">
        <Option/>
      </colorramp>
    </rasterrenderer>
    <brightnesscontrast gamma="1" brightness="0" contrast="0"/>
    <huesaturation grayscaleMode="0" colorizeStrength="100" saturation="0" colorizeRed="255" colorizeOn="0" colorizeGreen="128" invertColors="0" colorizeBlue="128"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
