from unittest import TestCase
from unittest.mock import MagicMock
import os.path
from greentea.text import Text
import torch.utils.data as ud
import limelight.dataset as d
import limelight.theme as t


class TestDataPointSource(TestCase):

    def test_read_text_non_utf8(self):
        dirname = os.path.dirname(__file__)
        data_point_id = d.DataPointId(51865)
        meta = d.DataPointMeta(
            data_point_id, t.Theme.COMP_SYS_MAC_HARDWARE)
        target = d.DataPointSource(dirname, meta)

        actual = target.read_text()
        self.assertIsInstance(
            actual,
            Text,
            "Ignore bytes that utf-8 codec can't decode")


class TestDataPointMeta(TestCase):

    def test_get_id_str(self):
        data_point_id = MagicMock(spec=d.DataPointId)
        theme = MagicMock(spec=t.Theme)
        target = d.DataPointMeta(data_point_id, theme)

        actual = target.get_id_str()

        self.assertEqual(actual, data_point_id.get_as_str.return_value)


class TestDataPoint(TestCase):

    def setUp(self):
        self.dataset = os.path.join(os.path.dirname(__file__),
                                    'test_data_point_sources.csv')

    def test_read_csv(self):
        sources = d.DataPointSources.read_csv(self.dataset)

        self.assertIsInstance(sources, d.DataPointSources)


class TestDataset(TestCase):

    def setUp(self):
        item0 = d.DataPointSource(
            'a', MagicMock(spec=d.DataPointMeta))
        item1 = d.DataPointSource(
            'b', MagicMock(spec=d.DataPointMeta))
        self.sources = d.DataPointSources([item0, item1])
        self.dataset = d.Dataset(self.sources, lambda x: x.directory)

    def test_transformer(self):
        self.assertEqual(list(self.dataset), ['a', 'b'])

    def test_loader(self):
        loader = ud.DataLoader(self.dataset, batch_size=2)
        self.assertEqual(list(loader), [['a', 'b']])
