/* Choose new high-water mark readings using the terrain coverage objective. */
const MarkRouteCrew = (() => {
  function plan(sites, budget, candidateIndices, targetIndices) {
    const withheld = sites.flatMap((site, index) => site.surveyed ? [] : [index]);
    const candidates = candidateIndices ?? withheld;
    const targets = targetIndices ?? withheld;
    if (!sites.length || !candidates.length || !targets.length || budget <= 0) return [];

    const meanLat = sites.reduce((sum, point) => sum + point.lat, 0) / sites.length;
    const horizontal = 111.2 * Math.cos(meanLat * Math.PI / 180);
    const distance = (a, b) => Math.hypot(horizontal * (a.lon - b.lon), 111.2 * (a.lat - b.lat));

    const groups = new Map();
    for (const site of sites) {
      if (!groups.has(site.environment)) groups.set(site.environment, []);
      if (site.surveyed) groups.get(site.environment).push(site);
    }
    const measured = sites.filter(site => site.surveyed);
    if (!measured.length) return [];
    const rows = candidates.map(index => ({index, site: sites[index], values: new Float64Array(targets.length)}));
    const coverage = new Float64Array(targets.length);
    for (const [environment, localAnchors] of groups) {
      const anchors = localAnchors.length ? localAnchors : measured;
      const localTargets = targets.flatMap((index, slot) => sites[index].environment === environment ? [slot] : []);
      const localRows = rows.filter(row => row.site.environment === environment);
      if (!anchors.length || !localTargets.length || !localRows.length) continue;

      const waiting = [...new Set([...localRows.map(row => row.index), ...localTargets.map(slot => targets[slot])])];
      const nearest = waiting.map(index => Math.min(...anchors.map(anchor => distance(sites[index], anchor)))).sort((a,b)=>a-b);
      const middle = Math.floor(nearest.length / 2);
      const median = nearest.length % 2 ? nearest[middle] : (nearest[middle - 1] + nearest[middle]) / 2;
      const scale = Math.max(.25, .75 * median);
      function similarity(a, b) {
        if (a === b) return 1;
        const d = Math.sqrt(3) * Math.hypot(distance(a, b) / scale, (a.groundFt - b.groundFt) / 45);
        return (1 + d) * Math.exp(-d) * (a.waterbody && a.waterbody === b.waterbody ? 1 : .35);
      }
      for (const slot of localTargets) {
        const query = sites[targets[slot]];
        coverage[slot] = Math.max(...anchors.map(anchor => similarity(query, anchor)));
        for (const row of localRows) row.values[slot] = similarity(query, row.site);
      }
    }
    const selected = [];
    while (selected.length < Math.min(budget, candidates.length)) {
      let best = null;
      for (const row of rows) {
        if (row.chosen) continue;
        let benefit = 0;
        let reached = 0;
        for (let i = 0; i < coverage.length; i++) {
          const increase = row.values[i] - coverage[i];
          if (increase > 0) {
            benefit += increase;
            if (increase >= .01 && targets[i] !== row.index) reached++;
          }
        }
        if (!best || benefit > best.benefit) best = {row, benefit, reached};
      }
      if (!best) break;
      const {row, reached} = best;
      row.chosen = true;
      for (let i = 0; i < coverage.length; i++) coverage[i] = Math.max(coverage[i], row.values[i]);
      selected.push({index: row.index, id: row.site.id, reached});
    }
    return selected;
  }

  function csv(header, rows) {
    const quote = (value) => {
      const str = String(value);
      const cell = typeof value !== 'number' && /^[\t\r \n]*[=+\-@]/.test(str) ? "'" + str : str;
      return '"' + cell.replace(/"/g, '""') + '"';
    };
    return [header.join(','), ...rows.map(row => row.map(quote).join(','))].join('\r\n') + '\r\n';
  }

  function queueCsv(picks, sites) {
    return csv(['priority', 'site_id', 'latitude', 'longitude', 'environment', 'waterbody'],
      picks.map(({index}, order) => {
        const site = sites[index];
        return [order + 1, site.id, site.lat, site.lon, site.environment, site.waterbody];
      }));
  }

  function surveyCsv(sites) {
    return csv(['site_id', 'latitude', 'longitude', 'environment', 'waterbody',
      'terrain_ft', 'terrain_median_ft', 'relief_ft', 'elev_ft'],
      sites.map(site => [site.id, site.lat, site.lon, site.environment, site.waterbody, site.pointFt,
        site.groundFt, site.reliefFt, site.surveyed ? site.actualFt : '']));
  }

  return {plan, queueCsv, surveyCsv};
})();
