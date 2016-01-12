#-*- coding: utf-8 -*-


class InsarPair:
    """
    InsarPair object for a single InSAR pair.
    """

    def __init__(self, dates):
        """
        Init for InsarPair.

        Parameters
        ----------
        dates: array_like
            List of datetime objects corresponding to [masterdate, slavedate].
        """
        self.masterDate, self.slaveDate = dates
        return


# end of file
