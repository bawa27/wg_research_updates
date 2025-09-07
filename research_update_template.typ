#import "@preview/cetz:0.2.2": *

// Page setup with header containing images
#set page(
  paper: "a4",
  margin: (top: 3.5cm, bottom: 2cm, left: 2cm, right: 2cm),
  header: [
    #grid(
      columns: (1fr, 1fr),
      align: (left, right),
      gutter: 1em,
      [
        #image("images/qisD_pic.png", width: 3cm)
      ],
      [
        #image("images/wg_pig.svg", width: 4cm)
      ]
    )
    #line(length: 100%, stroke: 0.5pt)
    #v(0.5em)
  ],
  footer: context [
    #line(length: 100%, stroke: 0.5pt)
    #v(0.3em)
    #align(center)[
      Research Update - #datetime.today().display("[month repr:long] [day], [year]") - Page #counter(page).display()
    ]
  ]
)

// Document styling
#set text(
  font: "Times New Roman",
  size: 11pt,
  lang: "en"
)

#set heading(numbering: "1.")

#set par(
  justify: true,
  leading: 0.65em,
  first-line-indent: 1.5em
)

// Title and metadata
#align(center)[
  #text(size: 18pt, weight: "bold")[
    Research Update
  ]
  
  #v(1em)
  
  #text(size: 14pt)[
    #datetime.today().display("[month repr:long] [day], [year]")
  ]
  
  #v(0.5em)
  
  #text(size: 12pt)[
    Aryan Bawa \
    Dartmouth College \
    Department of Physics and Astronomy
  ]
]

#v(2em)

// Abstract section
= Abstract

#lorem(100)

#v(1em)

// Main content sections
= Introduction

#lorem(150)

== Background

#lorem(120)

== Objectives

The main objectives of this research update are:

1. #lorem(20)
2. #lorem(25)
3. #lorem(30)

= Methodology

#lorem(200)

== Data Collection

#lorem(100)

== Analysis Framework

#lorem(80)

= Results

#lorem(180)

== Key Findings

#lorem(150)

=== Finding 1

#lorem(100)

=== Finding 2

#lorem(120)

== Statistical Analysis

#lorem(90)

= Discussion

#lorem(200)

== Implications

#lorem(100)

== Limitations

#lorem(80)

= Future Work

#lorem(100)

The next steps in this research include:

- #lorem(15)
- #lorem(20)
- #lorem(18)
- #lorem(25)

= Conclusion

#lorem(120)

= References

+ Author, A. (2024). "Title of Paper." _Journal Name_, 10(2), 123-145.

+ Author, B., & Author, C. (2023). "Another Important Paper." In _Conference Proceedings_ (pp. 67-89). Publisher.

+ Author, D. (2024). _Book Title_. Publisher Name.

#pagebreak()

= Appendix

== Additional Data

#lorem(100)

== Code Snippets

```python
# Example code
def analyze_data(data):
    """
    Analyze the research data
    """
    results = {}
    for item in data:
        # Process each item
        results[item.id] = process_item(item)
    return results
```

== Figures and Tables
