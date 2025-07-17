---
title: Functions
weight: 1
---

## Functions
-'''Dataset Loading (openmoa.dataset.*)'''
  -'''stream_loader()'''	Return a streaming dataset with an infinite iterator based on its name, and specify a feature drift strategy	'''ds = om.dataset.stream_loader('synthetic_open', n_samples=1e6, feature_pace=500)'''
  -'''file_stream(path, fmt='csv|jsonl|parquet')'''	Read real-time append files from local files or S3/HDFS	'''ds = om.dataset.file_stream('s3://bucket/log.parquet')'''
  -'''kafka_stream(topic, brokers, schema)'''	Directly consume from Kafka topics	'''ds = om.dataset.kafka_stream('iot-sensor', brokers='kafka:9092')'''
  -'''arxiv_open_citation_stream()'''	Open feature drift flow reserved interface for real academic graphs

-'''Pre-processing (openmoa.preprocess.*)'''
  -'''adaptive_standardize()'''	Online mean variance standardization, supporting cold start for new features	'''ds = om.preprocess.adaptive_standardize(ds, alpha=0.01)'''
  -'''drift_detector()''' Feature space drift detection (KL divergence/Maximum Mean Discrepancy)	'''flag = om.preprocess.drift_detector(window=1000)'''
  -'''feature_hashing(n_buckets)'''	Hash dimension reduction when feature space explodes	'''ds = om.preprocess.feature_hashing(ds, n_buckets=2**20)'''
  -'''missing_value_imputer(strategy='mean|median|zero')'''	Fill in missing values online	'''ds = om.preprocess.missing_value_imputer(strategy='zero')'''


## Folder Structure

There are **4 main folders for Hugo-based sites**:

- `content/` for your Markdown-formatted content files (homepage, etc.)
  - `_index.md` the homepage (**Hugo requires that the homepage and archive pages have an underscore prefix**)
- `assets/`
  - `media/` for your media files (images, videos)
    - `icons/custom/` upload any custom SVG icons you want to use
- `config/_default/` for your site configuration files
  - `hugo.yaml` to configure Hugo (site title, URL, Hugo options, setup per-folder page features)
  - `module.yaml` to install or uninstall Hugo themes and plugins
  - `params.yaml` to configure Hugo Blox options (SEO, analytics, site features)
  - `menus.yaml` to configure your menu links (if the menu is enabled in `params.yaml`)
  - `languages.yaml` to configure your site's language or to set language-specific options in a multilingual site
- `static/uploads/` for any files you want visitors to download, such as a PDF
- `go.mod` sets the version of Hugo themes/plugins which your site uses


## Hugo File Naming Convention

Hugo gives us two options to name standard page files: as `TITLE/index.md` or `TITLE.md` where `TITLE` is your page name.

The page name should be lowercase and using hyphens (`-`) instead of spaces.

Both approaches result in the same output, so you can choose your preferred approach to naming and organizing files. A benefit to the folder-based approach is that all your page's files (such as images) are self-contained within the page's folder, so it's more portable if you wish to share the original Markdown page with someone.

The homepage is a special case as **Hugo requires the homepage and listing pages to be named** `_index.md`.

## Docs Navigation

The docs navigation is automatically generated based on the content in the `docs/` folder and is sorted alphabetically.

The order of pages can be changed by adding the `weight` parameter in the front matter of your Markdown files.

In the example below, the `example.md` page will appear before the `test.md` page as it has a lower `weight`:

Page `example.md`:

```yaml
---
title: My Example
weight: 1
---
```

Page `test.md`:

```yaml
---
title: My Test
weight: 2
---
```
