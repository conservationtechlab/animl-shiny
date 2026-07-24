# ============================================================================
# animl-shiny
# ============================================================================
#
# Pipeline UI, built one step at a time:
#   Step 1: select a folder of camera trap images/videos, then run
#           animl::build_file_manifest() on it and preview the result.
#   Step 2: pick how many frames to pull per video, then run
#           animl::extract_frames() against that manifest.
#
# The manifest (a table of every image/video file found, plus EXIF metadata
# like timestamps) from Step 1 is the input Step 2 needs. Later pipeline
# steps -- MegaDetector, species classification -- will build on Step 2's
# output the same way.
#
# NOTE: saving intermediate outputs to disk (out_file / out_dir args,
# WorkingDirectory()'s save-file locations) is intentionally left out of
# these helpers for now -- that's a separate concern to design properly
# later, rather than something to bolt on ad hoc here.
#
# Required packages:
#   install.packages(c("shiny", "shinyFiles", "DT", "fs"))
#   install.packages("animl")   # or devtools::install_github("conservationtechlab/animl")
# ============================================================================

library(shiny)
library(shinyFiles)
library(DT)
library(fs)
library(animl)


# ============================================================================
# Helper functions
# ============================================================================
# Pulled out of the reactive blocks below and documented with roxygen2 tags
# so they read the same way animl-py's docstrings do -- what goes in, what
# comes out, and any side effects worth knowing about. RStudio can scaffold
# this skeleton for you: put your cursor inside a function and use
# Code -> Insert Roxygen Skeleton.

#' Set up animl's working directory for a folder of camera trap media
#'
#' Kept separate from build_manifest_from_dir() since it's really a
#' distinct setup step, not part of building the manifest itself.
#'
#' @param imagedir string, path to a folder of camera trap images/videos
#'
#' @details WorkingDirectory() does not return an object -- it assigns
#'   global variables into whatever environment you pass it. We pass
#'   globalenv() so those variables are available afterwards. We're not
#'   relying on any of those variables yet (see note at top of file re:
#'   saving intermediate outputs) -- this just satisfies animl's setup step.
set_working_directory <- function(imagedir) {
  WorkingDirectory(imagedir, globalenv())
}


#' Build a file manifest for a directory of camera trap media
#'
#' Thin wrapper around animl's build_file_manifest().
#'
#' @param imagedir string, path to a folder of camera trap images/videos
#'
#' @return data frame with one row per file found (FilePath, EXIF fields
#'   such as DateTimeOriginal, etc.)
build_manifest_from_dir <- function(imagedir) {
  set_working_directory(imagedir)
  build_file_manifest(imagedir, exif = TRUE)
}


#' Extract still frames from the videos in a file manifest
#'
#' Thin wrapper around animl's extract_frames().
#'
#' @param files data frame, the manifest produced by build_manifest_from_dir()
#' @param frames_per_video integer, number of frames to sample per video
#'
#' @return data frame of still frames (one row per extracted frame), ready
#'   to be fed into MegaDetector in a later pipeline step
extract_video_frames <- function(files, frames_per_video) {
  extract_frames(
    files,
    frames      = frames_per_video,
    parallel    = TRUE,
    num_workers = parallel::detectCores()
  )
}


# ============================================================================
# UI -- defines what the user SEES. No logic runs here, just layout.
# ============================================================================
#
# fluidPage() is the standard Shiny page container -- responsive width,
# Bootstrap styling out of the box. Note the capital P: fluidPage(), not
# fluidpage() -- R function names are case-sensitive.
ui <- fluidPage(

  titlePanel("AniML Camera Trap Manifest Builder"),

  # sidebarLayout() gives us the classic two-column Shiny layout:
  # a narrow sidebarPanel() for controls, and a wider mainPanel() for output.
  sidebarLayout(

    sidebarPanel(

      # ---- Step 1: folder selection + manifest controls -----------------
      h4("Step 1: Build File Manifest"),

      # shinyDirButton() draws a button that, when clicked, opens a folder
      # browser dialog (server-side, so it works even if this app is
      # deployed to a browser and not just run locally in RStudio).
      # id = "dir" is how we'll refer to whatever gets picked, on the
      # server side, as input$dir.
      shinyDirButton(
        id = "dir",
        label = "Select Image Folder",
        title = "Choose a folder containing camera trap images/videos"
      ),

      br(), br(),

      # verbatimTextOutput() is the UI-side placeholder for text we'll
      # generate on the server with renderPrint(). The "dirpath" id here
      # must match output$dirpath in the server function below --
      # every render*() / *Output() pair is linked by a matching id string.
      verbatimTextOutput("dirpath"),

      br(),

      # actionButton() just counts clicks -- every time it's clicked,
      # input$run increments by 1. It doesn't do anything by itself;
      # the server watches for that increment (see eventReactive below).
      actionButton("run", "Build File Manifest", class = "btn-primary"),

      hr(),

      helpText(
        "Selects a directory, then calls animl::build_file_manifest() ",
        "on it (with exif = TRUE) and displays the resulting manifest."
      ),

      # downloadButton() pairs with downloadHandler() on the server --
      # clicking it triggers a file save dialog in the browser.
      downloadButton("download_manifest", "Download Manifest (CSV)"),

      hr(),

      # ---- Step 2: frame extraction controls -------------------------
      h4("Step 2: Extract Frames"),

      # numericInput() gives the user a plain number field (with up/down
      # arrows) instead of a slider -- a good fit here since "frames per
      # video" is a small, precise integer rather than a range to explore.
      numericInput(
        inputId = "frames_per_video",
        label   = "Frames to pull per video",
        value   = 3,     # sensible default, matches the animl README example
        min     = 1,
        max     = 20,
        step    = 1
      ),

      actionButton("extract", "Extract Frames", class = "btn-primary"),

      helpText(
        "Runs animl::extract_frames() on the manifest above, pulling the ",
        "chosen number of still frames from each video for classification. ",
        "Requires Step 1 (Build File Manifest) to have run first."
      )
    ),

    mainPanel(
      # A single table that reflects whichever step has run most
      # recently: the file manifest after Step 1, then updated in place
      # to show extracted frames after Step 2 -- rather than stacking a
      # second table below it.
      textOutput("status"),
      DTOutput("results_table")
    )
  )
)


# ============================================================================
# SERVER -- defines what the app DOES. Runs once per user session.
# ============================================================================
#
# input   = everything the user has clicked/typed/selected in the UI
# output  = everything we render back to the UI (must match *Output() ids)
# session = info/hooks tied to this specific browser connection
server <- function(input, output, session) {

  # ---- Folder browsing setup ----------------------------------------------
  # shinyFiles needs to know which top-level "roots" it's allowed to browse
  # from (for security -- you don't want a web app browsing a server's
  # entire filesystem by default). We offer the user's home folder, the
  # R installation folder, and whatever drives shinyFiles auto-detects
  # (C:\, D:\, etc. on Windows).
  volumes <- c(
    Home = fs::path_home(),
    "R Installation" = R.home(),
    shinyFiles::getVolumes()()
  )

  # This wires up input$dir to actually respond to shinyDirButton clicks,
  # using the roots we just defined.
  shinyDirChoose(input, "dir", roots = volumes, session = session)

  # ---- Reactive: the currently selected folder ----------------------------
  # reactive() creates a value that automatically recalculates whenever
  # something it depends on (here, input$dir) changes. Think of it like a
  # lazy, auto-updating variable. We call it as selected_dir() elsewhere,
  # like calling a function, even though it behaves like reactive data.
  selected_dir <- reactive({
    # req() is a guard clause: if input$dir isn't set yet (user hasn't
    # picked a folder), stop here quietly instead of throwing an error.
    req(input$dir)

    # shinyFiles gives us back a compact internal representation of the
    # chosen path; parseDirPath() converts it into an actual usable
    # file system path string.
    parseDirPath(volumes, input$dir)
  })

  # ---- Output: show which folder is selected ------------------------------
  # renderPrint() captures whatever gets printed/cat()'d and sends it to
  # the matching verbatimTextOutput("dirpath") in the UI.
  output$dirpath <- renderPrint({
    if (is.integer(input$dir)) {
      # input$dir starts out as an integer placeholder before any
      # selection has been made -- this is the "nothing picked yet" state.
      cat("No folder selected yet")
    } else {
      cat("Selected:", selected_dir())
    }
  })

  # ---- Reactive: the file manifest, built only on button click ------------
  # eventReactive() is like reactive(), but it only recalculates when a
  # SPECIFIC trigger fires -- here, input$run (the action button) -- rather
  # than any time any dependency changes. This is what makes the manifest
  # build "on demand" instead of re-running every time selected_dir()
  # changes (e.g. while the user is still browsing folders).
  manifest <- eventReactive(input$run, {
    req(selected_dir())  # don't run if no folder has been picked

    # withProgress()/incProgress() show a progress bar in the UI while
    # this block runs -- purely cosmetic, doesn't affect the logic.
    withProgress(message = "Building file manifest...", value = 0.3, {
      files <- build_manifest_from_dir(selected_dir())
      incProgress(0.7)
      files  # eventReactive returns whatever the block's last line is
    })
  })

  # ---- Reactive: the single table shown in the UI --------------------------
  # results_data() holds whatever should currently be displayed:
  #   - the manifest, once Step 1 has run
  #   - then the extracted frames, once Step 2 has also run
  # reactiveVal() is a plain mutable reactive value (unlike reactive()/
  # eventReactive(), which derive their value from a formula) -- we update
  # it explicitly with observeEvent() below whenever a step completes.
  results_data <- reactiveVal(NULL)

  observeEvent(input$run, {
    results_data(manifest())
  })

  observeEvent(input$extract, {
    req(manifest())  # Step 2 requires Step 1 to have already run

    withProgress(message = "Extracting frames...", value = 0.2, {
      allframes <- extract_video_frames(manifest(), input$frames_per_video)
      incProgress(0.8)
      results_data(allframes)  # replaces the manifest in the same table
    })
  })

  # ---- Output: summary line -------------------------------------------------
  output$status <- renderText({
    req(results_data())
    n <- nrow(results_data())
    if (n == 0) {
      "No rows to show yet."
    } else {
      paste0("Showing ", n, " row(s).")
    }
  })

  # ---- Output: the interactive table ---------------------------------------
  # renderDT() / DTOutput() is the DT-package equivalent of
  # renderTable()/tableOutput(), but with sorting, searching, and paging.
  output$results_table <- renderDT({
    req(results_data())
    datatable(results_data(), options = list(scrollX = TRUE, pageLength = 15))
  })

  # ---- Output: CSV download ---------------------------------------------
  # downloadHandler() needs two pieces:
  #   filename -> what the saved file should be called
  #   content  -> a function that writes the data to the temp `file` path
  #               Shiny hands it; Shiny then streams that file to the user
  output$download_manifest <- downloadHandler(
    filename = function() "file_manifest.csv",
    content = function(file) {
      req(results_data())
      write.csv(results_data(), file, row.names = FALSE)
    }
  )
}

# ============================================================================
# Launch the app by combining the ui and server pieces defined above.
# This is the line RStudio's "Run App" button actually executes.
# ============================================================================
shinyApp(ui, server)
