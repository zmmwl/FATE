
    def save_summary(self):
        self.tracker.log_component_summary(summary_data=self.summary())

    def summary(self):
        return copy.deepcopy(self._summary)

    def set_summary(self, new_summary):
        """
        Model summary setter
        Parameters
        ----------
        new_summary: dict, summary to replace the original one

        Returns
        -------

        """

        if not isinstance(new_summary, dict):
            raise ValueError(
                f"summary should be of dict type, received {type(new_summary)} instead."
            )
        self._summary = copy.deepcopy(new_summary)

    def add_summary(self, new_key, new_value):
        """
        Add key:value pair to model summary
        Parameters
        ----------
        new_key: str
        new_value: object

        Returns
        -------

        """

        original_value = self._summary.get(new_key, None)
        if original_value is not None:
            LOGGER.warning(
                f"{new_key} already exists in model summary."
                f"Corresponding value {original_value} will be replaced by {new_value}"
            )
        self._summary[new_key] = new_value
        # LOGGER.debug(f"{new_key}: {new_value} added to summary.")

    def merge_summary(self, new_content, suffix=None, suffix_sep="_"):
        """
        Merge new content into model summary
        Parameters
        ----------
        new_content: dict, content to be merged into summary
        suffix: str or None, suffix used to create new key if any key in new_content already exixts in model summary
        suffix_sep: string, default '_', suffix separator used to create new key

        Returns
        -------

        """

        if not isinstance(new_content, dict):
            raise ValueError(
                f"To merge new content into model summary, "
                f"value must be of dict type, received {type(new_content)} instead."
            )
        new_summary = self.summary()
        keyset = new_summary.keys() | new_content.keys()
        for key in keyset:
            if key in new_summary and key in new_content:
                if suffix is not None:
                    new_key = f"{key}{suffix_sep}{suffix}"
                else:
                    new_key = key
                new_value = new_content.get(key)
                new_summary[new_key] = new_value
            elif key in new_content:
                new_summary[key] = new_content.get(key)
            else:
                pass
        self.set_summary(new_summary)
