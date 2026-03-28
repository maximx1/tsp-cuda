function generateCppMatrixString(n) {
  if (n <= 0) return 'int routes[0][0] = { };';
  const rows = [];
  for (let i = 0; i < n; i++) {
    const cells = [];
    for (let j = 0; j < n; j++) {
      let v;
      if (i === j) {
        v = -1;
      } else if (j > i) {
        v = Math.floor(Math.random() * 99) + 1; // 1..99
      } else {
        v = Number(rows[j].split(/,\s*/)[i]);
      }
      cells.push(String(v).padStart(2, ' '));
    }
    rows[i] = cells.join(', ');
  }

  const header = `int routes[${n}][${n}] = {\n`;
  const body = rows.map(r => `    { ${r} }`).join(',\n');
  const footer = `\n};`;
  return header + body + footer;
}

const text = generateCppMatrixString(30);
console.log(text);