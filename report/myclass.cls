
\NeedsTeXFormat{LaTeX2e}

%    ****************************************
%    *               OPTIONS                *
%    ****************************************
%
% Option values are now declared first thing.  Defaults are set to
% 12 point, onesided, phd (i.e., not masters), final (i.e., not draft) mode
% Only the phd/masters/oldmasters options are handled in this file (other
% than changing default values).

\newcommand\@masters{}
\DeclareOption{phd}{\renewcommand\@masters{0}}
\DeclareOption{masters}{\renewcommand\@masters{1}}
\DeclareOption{oldmasters}{\renewcommand\@masters{2}}
\ExecuteOptions{phd}

%  The Graduate Division requirements essentially require that theses be
%  formatted as if the oneside option had been used.  In the memoir class,
%  however, twoside is the default.  The following code relies on the memoir
%  class for oneside/twoside formatting, but changes the default to oneside.
%  Ideas for this code were adapted from an answer on stackexchange,
%    http://tex.stackexchange.com/questions/12933/new-options-and-loaded-class-options-bug
%  (accessed on 12 September 2011).

\newcommand\thes@sidedness{-1}
\DeclareOption{oneside}
  {\renewcommand\thes@sidedness{1}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{twoside}
  {\renewcommand\thes@sidedness{2}\PassOptionsToClass{\CurrentOption}{memoir}}

%  Changing the default point size from 10pt to 12pt uses a similar procedure,
%  but it's messier.

\newcommand\thes@ptsize{0}
\DeclareOption{9pt}
  {\renewcommand\thes@ptsize{9}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{10pt}
  {\renewcommand\thes@ptsize{10}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{11pt}
  {\renewcommand\thes@ptsize{11}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{12pt}
  {\renewcommand\thes@ptsize{12}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{14pt}
  {\renewcommand\thes@ptsize{14}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{17pt}
  {\renewcommand\thes@ptsize{17}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{20pt}
  {\renewcommand\thes@ptsize{20}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{25pt}
  {\renewcommand\thes@ptsize{25}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{30pt}
  {\renewcommand\thes@ptsize{30}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{36pt}
  {\renewcommand\thes@ptsize{36}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{48pt}
  {\renewcommand\thes@ptsize{48}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{60pt}
  {\renewcommand\thes@ptsize{60}\PassOptionsToClass{\CurrentOption}{memoir}}
\DeclareOption{*pt}
  {\renewcommand\thes@ptsize{99}\PassOptionsToClass{\CurrentOption}{memoir}}

% Other options are passed on to the memoir class.

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}

% The \ProcessOptions command causes the execution of every option command
% FOO which is declared and for which the user typed the FOO option in his
% \documentclass.  For every undeclared option BAR there will be a warning
% message. (BBF 10/31/94)

\ProcessOptions\relax

% Take care of changed defaults

\ifnum\thes@sidedness<0
  \PassOptionsToClass{oneside}{memoir}
  \renewcommand\thes@sidedness{1}
\fi

\ifnum\thes@ptsize=0
  \PassOptionsToClass{12pt}{memoir}
  \renewcommand\thes@ptsize{12}
\fi

% Now load the memoir class, and configure it appropriately.

\LoadClass{memoir}

\setlrmarginsandblock{1in}{*}{*}
  \@tempdima=1in  \advance\@tempdima 25pt \advance\@tempdima\normalbaselineskip
  \@tempdimb=.75in  \advance\@tempdimb 30pt
\setulmarginsandblock{\@tempdima}{\@tempdimb}{*}

\AtBeginDocument{\checkandfixthelayout}

% The Grad. Division requires that "Placement of page numbers must be
% consistent throughout."

\pagestyle{simple}
\aliaspagestyle{book}{simple}
\aliaspagestyle{part}{simple}
\aliaspagestyle{chapter}{simple}
\aliaspagestyle{title}{simple}

% Issue warnings if the settings are not appropriate for a final copy

\ifnum\thes@sidedness=2\relax
  \ifdraftdoc\else
 ****** Use the twoside option only for drafts. \@spaces\@spaces ******}
\fi\fi

\ifnum\thes@ptsize<12\relax
  \ifdraftdoc\else
  {^^J\@spaces
 ****** Use the \thes@ptsize pt option only for drafts. \@spaces\@spaces
 ******^^J\@spaces
 ****** The Graduate Division requires 12-point type ******^^J\@spaces
 ****** or larger for the final version.\space\@spaces\@spaces\@spaces
 ******}
\fi\fi

% For some environments we switch back to single-spacing, also
% provides \dsp in case you need double-spacing where single-spacing
% is otherwise used.

\newcommand\ssp{\SingleSpacing}
\newcommand\dsp{\setSpacing{1.37}}

% Table of contents, lists of figures, and lists of tables should always
% be single-spaced.  If one wants them to be double-spaced instead (in a
% double-spaced document), the original macros are available as
% \memoirtableofcontents, \memoirlistoffigures, and \memoirlistoftables.

\let\memoirtableofcontents=\tableofcontents
\renewcommand\tableofcontents{\begin{SingleSpace}\memoirtableofcontents
  \end{SingleSpace}}

\let\memoirlistoffigures=\listoffigures
\renewcommand\listoffigures{\begin{SingleSpace}\memoirlistoffigures
  \end{SingleSpace}}

\let\memoirlistoftables=\listoftables
\renewcommand\listoftables{\begin{SingleSpace}\memoirlistoftables
  \end{SingleSpace}}

%  draft option (this is where the draft option is actually implemented)
%
\ifnum\@masters>1			% if oldmasters
  \ifdraftdoc \else
    \dsp				% Set to double spacing
  \fi
\fi

% \smallssp is used to produce tabular environments in the small font.
% This is required because single-spacing requires a change in font size.
% \scriptsizessp is a still smaller version of the same thing.
% Removed 09/2011

% \def\smallssp{\small\SingleSpacing}
% \def\scriptsizessp{\scriptsize\SingleSpacing}


%  Masters theses may not be called dissertations
%

\newcommand\@dissertation{\ifnum\@masters=0
  dissertation
\else
  thesis
\fi}


% PREPARING A FOREIGN LANGUAGE VERSION:
%
% This document class is for documents prepared in the English language.
% To prepare a version for another language, various English words must
% be replaced.  Many of the English words that require replacement are
% indicated in the documentation for the memoir class, currently in
% Section 18.20, Words and Phrases.
%
% Other English words that need replacement can be found in the macros
% supporting the title and approval pages.
%

% acknowledge environment:
\newcommand\acknowledgename{Acknowledgments}

%    ****************************************
%    *             FRONT MATTER             *
%    ****************************************
%

% DECLARATIONS
%
% These macros are used to declare arguments needed for the
% construction of the front matter.

% The year the degree will be officially conferred
\def\degreeyear#1{\gdef\@degreeyear{#1}}

% The semester (Fall or Spring) the degree will be officially conferred
\def\degreesemester#1{\gdef\@degreesemester{#1}}

% The full (unabbreviated) name of the degree
\def\degree#1{\gdef\@degree{#1}}

% The Joint Institution of your degree (optional, and rare)
\def\jointinstitution#1{\gdef\@jointinst{#1}}

% All previous degrees: one per line in chronological order
% This is no longer used, but is provided for backwards compatibility.
\def\prevdegrees#1{\gdef\@prevdegrees{#1}}

% The name of your committee's chair
%\def\chair#1{\gdef\@chair{#1}\gdef\@chairtitle{Chair}}

% The name of your committee's secondary chair (for chair + co-chair)
\def\cochair#1{\gdef\@cochair{#1}}

% The names of your committee's co-chairs (for two equal co-chairs)
%\def\cochairs#1#2{\gdef\@chair{#1}\gdef\@cochair{#2}\gdef\@chairtitle{Co-chair}}

% The names of your other committee members, one per line
%\def\othermembers#1{\gdef\@othermembers{#1}}

% The number of committee members
%\def\@numberofmembers{3}
%\def\numberofmembers#1{\gdef\@numberofmembers{#1}}

% The name of your degree's field (e.g. Psychology, Computer Science)
\def\field#1{\gdef\@field{#1}}

% The name of your degree's designated emphasis (optional, and rare)
\def\emphasis#1{\gdef\@emphasis{#1}}

% The name of your UoL Campus (e.g. Queen Mary, UCL, LSE)
\def\@campus{Berkeley}	% default value is Berkeley
\def\campus#1{\gdef\@campus{#1}}

% The alwayssingle environment ensures that front matter is always
% single column even in a double-column document.

\newenvironment{alwayssingle}
  {\if@twocolumn\@restonecoltrue\onecolumn \else \@restonecolfalse\newpage \fi}
  {\if@restonecol \twocolumn \else \newpage \fi}

% \maketitle outputs the complete titlepage.  It requires all the
% above macros.  It is probably not correct for joint (e.g. UCSD/SDSU) degrees.

% Set the font that will be used in the front matter headings
\def\fmfont{\fontsize\@xiipt{14.5}\selectfont}
\def\fmsmallfont{\fontsize\@xiipt{14pt}\selectfont}

\def\maketitle{
{\ssp
  \makeatletter
    \setlength{\@tempdima}{28pt}\advance\@tempdima-\baselineskip
    \parskip=\@tempdima minus 4pt
  \makeatother
\begin{alwayssingle}
  \hrule height 0pt
  % Fix things so that they're vertically centered
  % \@tempdima = 2 * ((\stockheight-\textheight) / 2 - \uppermargin) - \topskip
    \@tempdima=\stockheight
    \advance\@tempdima-\textheight
    \advance\@tempdima-2\uppermargin
    \advance\@tempdima-\topskip
    \vspace{\@tempdima}
  \vfill
  \let\footnotesize\small
  \let\footnoterule\relax
  \thispagestyle{empty}
  \setcounter{page}{1}

  \begin{center}
    \fmfont
    \textbf{\@title}\par
    \vspace{14pt minus 4pt}
    by \par
    {\@author}\par
    \vspace{28pt minus 8pt}
    A \@dissertation submitted in partial satisfaction of the \par
    requirements for the degree of \par
    \ifdefined\@jointinst
      Joint {\@degree}\\with {\@jointinst}\par
    \else
      {\textbf{\@degree}}\par
    \fi
    in \par
    {\@field}\par
    
    \vspace{30pt}
    Department of Computer Science \par
    \textbf{University College London} {\expandafter}\par
    \vspace{34pt minus 8pt}
    {
      \fmsmallfont
      %Committee in charge: \par
      %{\@chair}, \@chairtitle \\
      \ifdefined\@cochair
        {\@cochair}, Co-chair \\
      \fi
      %{\@othermembers}\par
      \vspace{14pt minus 4pt}
      \@degreesemester \ \@degreeyear\par
    }
  \end{center}
  \vfill
  \hrule height 0pt
\end{alwayssingle}

\setcounter{footnote}{0}
}}



% ABSTRACT
%
% The ABSTRACT environment allows for multi-page abstracts which,
% in accordance with UC rules, is numbered separately from the rest
% of the rest of the dissertation in Arabic.  It requires definition
% of the \title, \author, \degree, \field, \campus, and \chair macros.

\def\abstract{
\begin{alwayssingle}
% \pagestyle{simple}
\thispagestyle{simple}
\begin{center}
{\fmfont
{\bfseries \LARGE{\abstractname}}\par
}
\vspace{.1in}
\end{center}}

\def\endabstract{\par\vfil\null\end{alwayssingle}
}

% The abstract is no longer signed (12/2009)
%
% \def\abstractsignature{
% \hspace*{\fill}
% \begin{minipage}[t]{3.25in}
% \addtolength{\baselineskip}{-.5\baselineskip}
% \vspace{1.7cm}
% \rule{3.2in}{.5pt}\\
% \@chair\\
% Dissertation Committee Chair
% \end{minipage}}

% DEDICATION
%
% The dedication environment just makes sure the dedication gets its
% own page.

\newenvironment{dedication}
  {\begin{alwayssingle}}
  {\end{alwayssingle}}

% ACKNOWLEDGEMENTS
%
% The acknowledgements environment puts a large, bold, centered
% "Acknowledgements" label at the top of the page.

\newenvironment{acknowledgements}
  {\begin{alwayssingle}
    \begin{center}
    {\large \bfseries \acknowledgename}
    \end{center}
  }
  {\end{alwayssingle}}

% FRONTMATTER environment
%
% The FRONTMATTER environment makes sure that page numbering is set
% correctly (roman, lower-case, starting at 3) for the front matter
% that follows the abstract.  It also resets page-numbering for
% the remainder of the dissertation (arabic, starting at 1).

\renewenvironment{frontmatter}
  {\setcounter{page}{1}\renewcommand{\thepage}{\roman{page}}}
  {\newpage\renewcommand{\thepage}{\arabic{page}}\setcounter{page}{1}}


% Rename environments to allow them to switch back to double spacing

\ifnum\@masters>1

  % These were taken from memoir.cls 2010/09/19
  \newenvironment{memoirquotation}%
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item[]}%
               {\endlist}

  \newenvironment{memoirquote}%
               {\list{}{\rightmargin\leftmargin}%
                \item[]}%
               {\endlist}

  \newenvironment{memoirverse}[1][\linewidth]{%
    \refstepcounter{verse}%
    \setcounter{poemline}{0}\refstepcounter{poemline}%
    \setcounter{vslineno}{1}%
    \let\\=\@vscentercr
    \list{}{\itemsep     \z@
          \itemindent  -\vindent
          \listparindent\itemindent
          \leftmargin   \vleftmargin
          \parsep       \stanzaskip
          \ifdim #1<\linewidth%   %% short line
            \rightmargin        \z@
            \leftmargin         \linewidth
            \advance\leftmargin -#1\relax
            \advance\leftmargin -0.5\leftmargin
            \advance\leftmargin \vindent
          \else
            \ifdim #1>\linewidth%  %% long line
              \rightmargin \z@
              \leftmargin  \vindent
            \else%                   %% default
              \rightmargin \leftmargin
              \advance\leftmargin \vindent
            \fi
          \fi}
    \item[]}{\endlist}

  \ifdraftdoc\else

    \renewenvironment{quotation}%
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \SingleSpacing\item[]}%
               {\endlist}

    \renewenvironment{quote}%
               {\list{}{\rightmargin\leftmargin}%
                \SingleSpacing\item[]}%
               {\endlist}

    \renewenvironment{verse}[1][\linewidth]{%
      \refstepcounter{verse}%
      \setcounter{poemline}{0}\refstepcounter{poemline}%
      \setcounter{vslineno}{1}%
      \let\\=\@vscentercr
      \list{}{\itemsep     \z@
          \itemindent  -\vindent
          \listparindent\itemindent
          \leftmargin   \vleftmargin
          \parsep       \stanzaskip
          \ifdim #1<\linewidth%   %% short line
            \rightmargin        \z@
            \leftmargin         \linewidth
            \advance\leftmargin -#1\relax
            \advance\leftmargin -0.5\leftmargin
            \advance\leftmargin \vindent
          \else
            \ifdim #1>\linewidth%  %% long line
              \rightmargin \z@
              \leftmargin  \vindent
            \else%                   %% default
              \rightmargin \leftmargin
              \advance\leftmargin \vindent
            \fi
          \fi}
    \SingleSpacing\item[]}{\endlist}

  \fi
\fi

% smalltabular and smalltabular* are alternative tabular environments
% in the \small font.  They are required because single-spacing implies a
% change of font to a particular size.
% EVM, 7/20/93
% Removed 09/2011

% \def\smalltabular{\par\small\SingleSpacing\let\@halignto\@empty\@tabular}
% \let\endsmalltabular = \endtabular

% \@namedef{smalltabular*}#1{\par\small\SingleSpacing
%  \setlength\dimen@{#1}%
%    \edef\@halignto{to\the\dimen@}\@tabular}
% \expandafter \let \csname endsmalltabular*\endcsname = \endtabular

% \scriptsizetabular and \scriptsizetabular* are analogs of \smalltabular

% \def\scriptsizetabular{\par\scriptsize\SingleSpacing
%   \let\@halignto\@empty\@tabular}
% \let\endscriptsizetabular = \endtabular

% \@namedef{scriptsizetabular*}#1{\par\scriptsize\SingleSpacing
%  \setlength\dimen@{#1}%
%    \edef\@halignto{to\the\dimen@}\@tabular}
% \expandafter \let \csname endscriptsizetabular*\endcsname = \endtabular

\endinput
