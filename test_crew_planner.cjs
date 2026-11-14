/* Test the actual shipped planner against the independently fitted Python replay. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function csvRow(line) {
  const row = [];
  let value = '', wrapped = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      if (wrapped && line[i + 1] === '"') { value += '"'; i++; }
      else wrapped = !wrapped;
    } else if (char === ',' && !wrapped) {
      row.push(value);
      value = '';
    } else value += char;
  }
  row.push(value);
  return row;
}

function verifyCrewPlanner(packageDir, studyDir = packageDir) {
  const load = (file, directory = packageDir) => fs.readFileSync(path.join(directory, file), 'utf8');
  const source = load('crew_planner.js');
  const planner = vm.runInNewContext(`${source}\nMarkRouteCrew`, {});
  const {workedCase: example} = JSON.parse(load('crew_results.json', studyDir));
  const [header, ...lines] = load('remote_panel.csv').trimEnd().split(/\r?\n/);
  const names = header.split(',');
  const panel = lines.filter(line => line.startsWith(`${example.eventId},`)).map(line => {
    const data = Object.fromEntries(csvRow(line).map((column, index) => [names[index], column]));
    return {id: data.site_id, lat: Number(data.latitude), lon: Number(data.longitude), environment: data.environment,
      waterbody: data.waterbody, groundFt: Number(data.terrain_median_ft)};
  });
  assert.equal(panel.length, example.siteCount);
  const location = new Map(panel.map((p, i) => [p.id, i]));
  const queue = [];
  for (const cohort of example.cohorts) {
    const initial = new Set(cohort.initialSiteIds);
    const sites = panel.map(site => ({...site, surveyed: initial.has(site.id)}));
    const toIndex = ids => ids.map(id => location.get(id));
    const candidates = toIndex(cohort.candidateSiteIds);
    const targets = toIndex(cohort.scoringSiteIds);
    assert.ok([...candidates, ...targets].every(Number.isInteger));
    const picks = Array.from(planner.plan(sites, cohort.visitBudget, candidates, targets), row => row.id);
    assert.equal(picks.length, cohort.visitBudget);
    assert.ok(picks.every(id => cohort.candidateSiteIds.includes(id)));
    assert.ok(picks.every(id => !cohort.scoringSiteIds.includes(id) && !initial.has(id)));
    const poisoned = sites.map((site, index) => ({...site, actualFt: index * 1000000 - 1234567}));
    const repeated = Array.from(planner.plan(poisoned, cohort.visitBudget, candidates, targets), row => row.id);
    assert.deepEqual(repeated, picks);
    queue.push(...picks);
  }
  assert.deepEqual(queue, example.methods['terrain coverage'].chosenSiteIds);

  // The two identical candidates have equal positive gain, then no marginal gain.
  // The pinned batch policy uses stable candidate order and fills the entire budget.
  const fixture = [
    {id: 'initial', lat: 41.01, lon: -72, surveyed: true},
    {id: 'a', lat: 41, lon: -72, surveyed: false},
    {id: 'b', lat: 41, lon: -72, surveyed: false},
    {id: 'score', lat: 41, lon: -72, surveyed: false}
  ].map(site => ({...site, groundFt: 50, environment: 'Riverine', waterbody: 'river'}));
  const tiedQueue = Array.from(planner.plan(fixture, 2, [2, 1], [3]), row => row.id);
  assert.deepEqual(tiedQueue, ['b', 'a'], 'preserve explicit candidate order and fill the budget on zero residual gain');
  return {eventId: example.eventId, seed: example.seed, visits: queue.length, scoredSites: example.scoredCount,
    exactPythonOrder: true, surveyLabelsIgnored: true, tiedQueue};
}

if (require.main === module) console.log(JSON.stringify(verifyCrewPlanner(process.argv[2] ?? __dirname, process.argv[3])));
module.exports = verifyCrewPlanner;
