/* Test driver: evaluate a physics.js function and print the result as JSON.
 * Usage: node _node_check.js <functionName> '<json args array>'
 * physics.js lives inside the comfio package; resolved relative to this file. */
'use strict';
const path = require('path');
const P = require(path.join(
  __dirname, '..', '..', 'src', 'comfio', 'gui', 'tools',
  'comfort_tool', '_assets', 'physics.js'
));
const fn = process.argv[2];
const args = JSON.parse(process.argv[3] || '[]');
if (typeof P[fn] !== 'function') {
  console.error('unknown function: ' + fn);
  process.exit(2);
}
console.log(JSON.stringify(P[fn](...args)));
