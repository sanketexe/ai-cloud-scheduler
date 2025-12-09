import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Container,
  Typography,
  Paper,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  IconButton,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tabs,
  Tab,
  LinearProgress,
  Tooltip,
} from '@mui/material';
import { TreeView, TreeItem } from '@mui/lab';
import {
  ExpandMore,
  ChevronRight,
  Edit,
  Delete,
  Add,
  Refresh,
  CloudQueue,
  Storage,
  Computer,
  Category,
  Business,
  LocationOn,
  Group,
} from '@mui/icons-material';
import { migrationApi } from '../services/migrationApi';
import toast from 'react-hot-toast';

interface Resource {
  resource_id: string;
  resource_name: string;
  resource_type: string;
  provider: string;
  region: string;
  cost_monthly: number;
  tags: Record<string, string>;
  category?: string;
  team?: string;
  project?: string;
}

interface OrganizationalStructure {
  teams: string[];
  projects: string[];
  regions: string[];
  environments: string[];
}

interface HierarchyNode {
  id: string;
  name: string;
  type: string;
  children?: HierarchyNode[];
  resources?: Resource[];
  total_cost?: number;
}

const ResourceOrganization: React.FC = () => {
  const { projectId } = useParams<{ projectId: string }>();
  const [loading, setLoading] = useState(true);
  const [resources, setResources] = useState<Resource[]>([]);
  const [structure, setStructure] = useState<OrganizationalStructure | null>(null);
  const [hierarchy, setHierarchy] = useState<HierarchyNode[]>([]);
  const [selectedTab, setSelectedTab] = useState(0);
  const [selectedResources, setSelectedResources] = useState<string[]>([]);
  const [categorizeDialogOpen, setCategorizeDialogOpen] = useState(false);
  const [newCategory, setNewCategory] = useState('');
  const [newTeam, setNewTeam] = useState('');
  const [newProject, setNewProject] = useState('');

  useEffect(() => {
    if (projectId) {
      loadResources();
      loadStructure();
      loadHierarchy();
    }
  }, [projectId]);

  const loadResources = async () => {
    try {
      setLoading(true);
      const data = await migrationApi.getResources(projectId!);
      setResources(data as any);
    } catch (error) {
      toast.error('Failed to load resources');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const loadStructure = async () => {
    try {
      const data = await migrationApi.getOrganizationalStructure(projectId!);
      setStructure(data);
    } catch (error) {
      console.error('Failed to load organizational structure', error);
    }
  };

  const loadHierarchy = async () => {
    try {
      const data = await migrationApi.getResourceHierarchy(projectId!);
      setHierarchy(data);
    } catch (error) {
      console.error('Failed to load hierarchy', error);
    }
  };

  const handleCategorizeResources = async () => {
    try {
      await migrationApi.categorizeResources(projectId!, selectedResources, {
        category: newCategory,
        team: newTeam,
        project: newProject,
      });
      toast.success('Resources categorized successfully');
      setCategorizeDialogOpen(false);
      setSelectedResources([]);
      loadResources();
      loadHierarchy();
    } catch (error) {
      toast.error('Failed to categorize resources');
      console.error(error);
    }
  };

  const handleDiscoverResources = async () => {
    try {
      setLoading(true);
      await migrationApi.discoverResources(projectId!, 'aws', {});
      toast.success('Resource discovery started');
      setTimeout(() => {
        loadResources();
      }, 2000);
    } catch (error) {
      toast.error('Failed to start resource discovery');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const renderTreeItem = (node: HierarchyNode) => (
    <TreeItem
      key={node.id}
      nodeId={node.id}
      label={
        <Box sx={{ display: 'flex', alignItems: 'center', py: 1 }}>
          {node.type === 'team' && <Group sx={{ mr: 1 }} />}
          {node.type === 'project' && <Business sx={{ mr: 1 }} />}
          {node.type === 'region' && <LocationOn sx={{ mr: 1 }} />}
          {node.type === 'resource' && <CloudQueue sx={{ mr: 1 }} />}
          <Typography variant="body2" sx={{ flexGrow: 1 }}>
            {node.name}
          </Typography>
          {node.total_cost && (
            <Chip
              label={`$${node.total_cost.toFixed(2)}/mo`}
              size="small"
              color="primary"
            />
          )}
        </Box>
      }
    >
      {node.children?.map((child) => renderTreeItem(child))}
    </TreeItem>
  );

  if (loading) {
    return (
      <Container maxWidth="xl">
        <Box sx={{ mt: 4, textAlign: 'center' }}>
          <LinearProgress />
          <Typography sx={{ mt: 2 }}>Loading resources...</Typography>
        </Box>
      </Container>
    );
  }

  return (
    <Container maxWidth="xl">
      <Box sx={{ mt: 4, mb: 4 }}>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h4">Resource Organization</Typography>
          <Box>
            <Button
              variant="outlined"
              startIcon={<Refresh />}
              onClick={handleDiscoverResources}
              sx={{ mr: 2 }}
            >
              Discover Resources
            </Button>
            <Button
              variant="contained"
              startIcon={<Category />}
              onClick={() => setCategorizeDialogOpen(true)}
              disabled={selectedResources.length === 0}
            >
              Categorize Selected
            </Button>
          </Box>
        </Box>

        {/* Stats Cards */}
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="text.secondary">
                  Total Resources
                </Typography>
                <Typography variant="h3">{resources.length}</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="text.secondary">
                  Teams
                </Typography>
                <Typography variant="h3">{structure?.teams.length || 0}</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="text.secondary">
                  Projects
                </Typography>
                <Typography variant="h3">{structure?.projects.length || 0}</Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={3}>
            <Card>
              <CardContent>
                <Typography variant="h6" color="text.secondary">
                  Monthly Cost
                </Typography>
                <Typography variant="h3">
                  ${resources.reduce((sum, r) => sum + r.cost_monthly, 0).toFixed(0)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>

        {/* Tabs */}
        <Paper sx={{ mb: 3 }}>
          <Tabs value={selectedTab} onChange={(_, v) => setSelectedTab(v)}>
            <Tab label="Resource Inventory" />
            <Tab label="Hierarchy View" />
            <Tab label="Organizational Structure" />
          </Tabs>
        </Paper>

        {/* Tab Content */}
        {selectedTab === 0 && (
          <Paper>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell padding="checkbox">
                      <input
                        type="checkbox"
                        onChange={(e) => {
                          if (e.target.checked) {
                            setSelectedResources(resources.map((r) => r.resource_id));
                          } else {
                            setSelectedResources([]);
                          }
                        }}
                      />
                    </TableCell>
                    <TableCell>Resource Name</TableCell>
                    <TableCell>Type</TableCell>
                    <TableCell>Provider</TableCell>
                    <TableCell>Region</TableCell>
                    <TableCell>Category</TableCell>
                    <TableCell>Team</TableCell>
                    <TableCell>Project</TableCell>
                    <TableCell>Cost/Month</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {resources.map((resource) => (
                    <TableRow key={resource.resource_id}>
                      <TableCell padding="checkbox">
                        <input
                          type="checkbox"
                          checked={selectedResources.includes(resource.resource_id)}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setSelectedResources([...selectedResources, resource.resource_id]);
                            } else {
                              setSelectedResources(
                                selectedResources.filter((id) => id !== resource.resource_id)
                              );
                            }
                          }}
                        />
                      </TableCell>
                      <TableCell>{resource.resource_name}</TableCell>
                      <TableCell>
                        <Chip label={resource.resource_type} size="small" />
                      </TableCell>
                      <TableCell>{resource.provider}</TableCell>
                      <TableCell>{resource.region}</TableCell>
                      <TableCell>{resource.category || '-'}</TableCell>
                      <TableCell>{resource.team || '-'}</TableCell>
                      <TableCell>{resource.project || '-'}</TableCell>
                      <TableCell>${resource.cost_monthly.toFixed(2)}</TableCell>
                      <TableCell>
                        <Tooltip title="Edit">
                          <IconButton size="small">
                            <Edit fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        )}

        {selectedTab === 1 && (
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Resource Hierarchy
            </Typography>
            <TreeView
              defaultCollapseIcon={<ExpandMore />}
              defaultExpandIcon={<ChevronRight />}
            >
              {hierarchy.map((node) => renderTreeItem(node))}
            </TreeView>
          </Paper>
        )}

        {selectedTab === 2 && structure && (
          <Grid container spacing={3}>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Teams
                </Typography>
                {structure.teams.map((team) => (
                  <Chip key={team} label={team} sx={{ m: 0.5 }} />
                ))}
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Projects
                </Typography>
                {structure.projects.map((project) => (
                  <Chip key={project} label={project} sx={{ m: 0.5 }} />
                ))}
              </Paper>
            </Grid>
            <Grid item xs={12} md={4}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Regions
                </Typography>
                {structure.regions.map((region) => (
                  <Chip key={region} label={region} sx={{ m: 0.5 }} />
                ))}
              </Paper>
            </Grid>
          </Grid>
        )}

        {/* Categorize Dialog */}
        <Dialog open={categorizeDialogOpen} onClose={() => setCategorizeDialogOpen(false)}>
          <DialogTitle>Categorize Resources</DialogTitle>
          <DialogContent>
            <TextField
              fullWidth
              label="Category"
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value)}
              sx={{ mt: 2, mb: 2 }}
            />
            <TextField
              fullWidth
              label="Team"
              value={newTeam}
              onChange={(e) => setNewTeam(e.target.value)}
              sx={{ mb: 2 }}
            />
            <TextField
              fullWidth
              label="Project"
              value={newProject}
              onChange={(e) => setNewProject(e.target.value)}
            />
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setCategorizeDialogOpen(false)}>Cancel</Button>
            <Button onClick={handleCategorizeResources} variant="contained">
              Apply
            </Button>
          </DialogActions>
        </Dialog>
      </Box>
    </Container>
  );
};

export default ResourceOrganization;
